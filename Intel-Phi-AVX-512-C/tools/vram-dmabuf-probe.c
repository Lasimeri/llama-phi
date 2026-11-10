/* vram-dmabuf-probe [BUS [MIB]]: probe P1 of docs/research/phi-vram-dma.md.
 * On the GPU at PCI bus BUS (hex, default c7), allocate MIB MiB (default 64)
 * of device-local, host-visible memory (the GPU's BAR heap) exportable as a
 * dma-buf, write a pattern through the Vulkan mapping, export the dma-buf,
 * and check it: the fd's type and size, and whether a mapping of the dma-buf
 * itself shows the pattern (P1); then, when /dev/iommu can be opened, probe
 * P2: map the dma-buf into a fresh iommufd IOAS with IOMMU_IOAS_MAP_FILE at
 * IOVA 48 GiB and unmap it again. Documented interfaces only (Vulkan
 * VK_EXT_external_memory_dma_buf, VK_KHR_external_memory_fd; dma-buf mmap;
 * iommufd).
 * See vram-dmabuf-probe.md. */
#include <errno.h>
#include <fcntl.h>
#include <linux/iommufd.h>
#include <sys/ioctl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vulkan/vulkan.h>

#define CHECK(x)                                                                                         \
    do {                                                                                                 \
        VkResult r_ = (x);                                                                               \
        if (r_ != VK_SUCCESS) {                                                                          \
            printf("FAIL %s: VkResult %d\n", #x, (int)r_);                                               \
            return 1;                                                                                    \
        }                                                                                                \
    } while (0)

static uint32_t pat(size_t i) { return (uint32_t)(i * 2654435761u) ^ 0x5a5a5a5au; }

int main(int argc, char **argv)
{
    unsigned bus = argc > 1 ? (unsigned)strtoul(argv[1], NULL, 16) : 0xc7;
    VkDeviceSize size = (VkDeviceSize)(argc > 2 ? atoi(argv[2]) : 64) << 20;
    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
    VkInstance inst;
    CHECK(vkCreateInstance(&ici, NULL, &inst));

    uint32_t n = 0;
    CHECK(vkEnumeratePhysicalDevices(inst, &n, NULL));
    VkPhysicalDevice pds[16], pd = VK_NULL_HANDLE;
    if (n > 16)
        n = 16;
    CHECK(vkEnumeratePhysicalDevices(inst, &n, pds));
    for (uint32_t i = 0; i < n; i++) {
        VkPhysicalDevicePCIBusInfoPropertiesEXT pci = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT};
        VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &pci};
        vkGetPhysicalDeviceProperties2(pds[i], &p2);
        printf("device %u: %s, PCI %04x:%02x:%02x.%x\n", i, p2.properties.deviceName, pci.pciDomain, pci.pciBus,
               pci.pciDevice, pci.pciFunction);
        if (pci.pciBus == bus)
            pd = pds[i];
    }
    if (pd == VK_NULL_HANDLE) {
        printf("FAIL: no device on bus %02x\n", bus);
        return 1;
    }

    /* The BAR heap: device-local, host-visible, host-coherent, under 1 GiB. */
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    int type = -1;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        uint32_t h = mp.memoryTypes[i].heapIndex;
        if ((mp.memoryTypes[i].propertyFlags & want) == want && mp.memoryHeaps[h].size < (1ull << 30)) {
            type = (int)i;
            printf("BAR heap: memory type %u, heap %u, %llu MiB\n", i, h,
                   (unsigned long long)(mp.memoryHeaps[h].size >> 20));
            break;
        }
    }
    if (type < 0) {
        printf("FAIL: no device-local host-visible heap under 1 GiB\n");
        return 1;
    }

    VkPhysicalDeviceExternalBufferInfo ebi = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
    VkExternalBufferProperties ebp = {.sType = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES};
    vkGetPhysicalDeviceExternalBufferProperties(pd, &ebi, &ebp);
    VkExternalMemoryFeatureFlags f = ebp.externalMemoryProperties.externalMemoryFeatures;
    printf("dma-buf buffers: exportable %d, importable %d, dedicated only %d\n",
           !!(f & VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT), !!(f & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT),
           !!(f & VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT));
    if (!(f & VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT)) {
        printf("FAIL: the driver does not export buffers as dma-buf\n");
        return 1;
    }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = 0,
                                   .queueCount = 1, .pQueuePriorities = &prio};
    const char *ext[] = {VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME, VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME};
    VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1,
                              .pQueueCreateInfos = &qci, .enabledExtensionCount = 2, .ppEnabledExtensionNames = ext};
    VkDevice dev;
    CHECK(vkCreateDevice(pd, &dci, NULL, &dev));

    VkExternalMemoryBufferCreateInfo embci = {.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
                                              .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
    VkBufferCreateInfo bci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .pNext = &embci, .size = size,
                              .usage = ebi.usage, .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VkBuffer buf;
    CHECK(vkCreateBuffer(dev, &bci, NULL, &buf));
    VkMemoryDedicatedAllocateInfo ded = {.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, .buffer = buf};
    VkExportMemoryAllocateInfo emai = {.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO, .pNext = &ded,
                                       .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
    VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &emai,
                                .allocationSize = size, .memoryTypeIndex = (uint32_t)type};
    VkDeviceMemory mem;
    CHECK(vkAllocateMemory(dev, &mai, NULL, &mem));
    CHECK(vkBindBufferMemory(dev, buf, mem, 0));

    uint32_t *p;
    CHECK(vkMapMemory(dev, mem, 0, size, 0, (void **)&p));
    for (size_t i = 0; i < size / 4; i++)
        p[i] = pat(i);
    printf("wrote %llu MiB through the Vulkan mapping\n", (unsigned long long)(size >> 20));

    PFN_vkGetMemoryFdKHR getfd = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(dev, "vkGetMemoryFdKHR");
    VkMemoryGetFdInfoKHR gfi = {.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR, .memory = mem,
                                .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
    int fd = -1;
    CHECK(getfd(dev, &gfi, &fd));
    char link[64] = {0}, path[64];
    snprintf(path, sizeof path, "/proc/self/fd/%d", fd);
    ssize_t ln = readlink(path, link, sizeof link - 1);
    if (ln > 0)
        link[ln] = 0;
    off_t dsize = lseek(fd, 0, SEEK_END);
    printf("dma-buf fd %d: %s, size %lld bytes\n", fd, link, (long long)dsize);

    void *m = mmap(NULL, (size_t)size, PROT_READ, MAP_SHARED, fd, 0);
    if (m == MAP_FAILED) {
        printf("dma-buf mmap: refused (%s)\n", strerror(errno));
    } else {
        const uint32_t *q = m;
        size_t bad = 0;
        for (size_t i = 0; i < size / 4; i++)
            bad += q[i] != pat(i);
        printf("dma-buf mmap: %zu of %llu words differ from the pattern\n", bad, (unsigned long long)(size / 4));
        munmap(m, (size_t)size);
    }
    int ifd = open("/dev/iommu", O_RDWR);
    if (ifd < 0) {
        printf("P2 not run: /dev/iommu: %s (the phi group needs access, docs/research/phi-vram-dma.md)\n", strerror(errno));
    } else {
        struct iommu_ioas_alloc a = {.size = sizeof a};
        if (ioctl(ifd, IOMMU_IOAS_ALLOC, &a) != 0) {
            printf("P2 FAIL: IOMMU_IOAS_ALLOC: %s\n", strerror(errno));
        } else {
            struct iommu_ioas_map_file mf = {.size = sizeof mf,
                                              .flags = IOMMU_IOAS_MAP_FIXED_IOVA | IOMMU_IOAS_MAP_READABLE,
                                              .ioas_id = a.out_ioas_id, .fd = fd, .start = 0, .length = size,
                                              .iova = 48ull << 30};
            if (ioctl(ifd, IOMMU_IOAS_MAP_FILE, &mf) != 0) {
                printf("P2 FAIL: IOMMU_IOAS_MAP_FILE of the dma-buf: %s\n", strerror(errno));
            } else {
                printf("P2 PASS: iommufd mapped the dma-buf at IOVA 0x%llx, %llu MiB\n",
                       (unsigned long long)mf.iova, (unsigned long long)(size >> 20));
                struct iommu_ioas_unmap u = {.size = sizeof u, .ioas_id = a.out_ioas_id, .iova = mf.iova, .length = size};
                if (ioctl(ifd, IOMMU_IOAS_UNMAP, &u) != 0)
                    printf("P2: IOMMU_IOAS_UNMAP: %s\n", strerror(errno));
            }
            struct iommu_destroy d = {.size = sizeof d, .id = a.out_ioas_id};
            ioctl(ifd, IOMMU_DESTROY, &d);
        }
        close(ifd);
    }
    close(fd);
    vkUnmapMemory(dev, mem);
    vkFreeMemory(dev, mem, NULL);
    vkDestroyBuffer(dev, buf, NULL);
    vkDestroyDevice(dev, NULL);
    vkDestroyInstance(inst, NULL);
    printf("P1 %s\n", (dsize == (off_t)size && strstr(link, "dmabuf")) ? "PASS: a dma-buf of the BAR heap was exported"
                                                                         : "FAIL");
    return 0;
}
