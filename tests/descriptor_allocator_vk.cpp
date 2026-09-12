#include <vulkan/vulkan.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

namespace stl = std;
#include "../src/descriptor_allocator_vk.h"

using Allocator = bgfx::vk::DescriptorAllocatorVK;

static void require(bool condition, const char* message)
{
	if (!condition)
	{
		std::fprintf(stderr, "FAIL: %s\n", message);
		std::exit(1);
	}
}

static void check(VkResult result)
{
	if (VK_SUCCESS != result)
	{
		std::fprintf(stderr, "Unexpected VkResult: %d\n", result);
		std::exit(1);
	}
}

struct Mock
{
	static inline std::map<VkDescriptorPool, uint32_t> pools;
	static inline uint32_t created = 0;
	static inline uint32_t resetCount = 0;
	static inline VkResult createResult = VK_SUCCESS;
	static inline VkResult allocateResult = VK_SUCCESS;
	static inline VkResult resetResult = VK_SUCCESS;
	static inline uint32_t capacity = 1024;
	static inline bool failOnce = false;

	static VKAPI_ATTR VkResult VKAPI_CALL create(VkDevice, const VkDescriptorPoolCreateInfo* info,
		const VkAllocationCallbacks*, VkDescriptorPool* pool)
	{
		require(info->flags == 0 && info->maxSets == 1024, "pool is reset as a whole");
		if (createResult != VK_SUCCESS) return createResult;
		*pool = (VkDescriptorPool)(uintptr_t)++created;
		pools.emplace(*pool, 0);
		return VK_SUCCESS;
	}

	static VKAPI_ATTR void VKAPI_CALL destroy(VkDevice, VkDescriptorPool pool, const VkAllocationCallbacks*)
	{
		require(pools.erase(pool) == 1, "destroy owned pool exactly once");
	}

	static VKAPI_ATTR VkResult VKAPI_CALL reset(VkDevice, VkDescriptorPool pool, VkDescriptorPoolResetFlags)
	{
		if (resetResult != VK_SUCCESS) return resetResult;
		pools.at(pool) = 0;
		++resetCount;
		return VK_SUCCESS;
	}

	static VKAPI_ATTR VkResult VKAPI_CALL allocate(VkDevice, const VkDescriptorSetAllocateInfo* info, VkDescriptorSet* set)
	{
		*set = VK_NULL_HANDLE;
		if (allocateResult != VK_SUCCESS)
		{
			const auto result = allocateResult;
			if (failOnce) allocateResult = VK_SUCCESS;
			return result;
		}
		auto& used = pools.at(info->descriptorPool);
		if (used == capacity) return VK_ERROR_OUT_OF_POOL_MEMORY;
		*set = (VkDescriptorSet)(uintptr_t)(++used);
		return VK_SUCCESS;
	}

	static Allocator make()
	{
		Allocator allocator;
		allocator.init({VK_NULL_HANDLE, nullptr, create, destroy, reset, allocate, 16});
		return allocator;
	}
};

static void allocateMany(Allocator& allocator, uint32_t count)
{
	for (uint32_t i = 0; i < count; ++i)
	{
		VkDescriptorSet set = VK_NULL_HANDLE;
		check(allocator.allocate(VK_NULL_HANDLE, &set));
		require(set != VK_NULL_HANDLE, "successful allocation has a set");
	}
}

static void unitTests()
{
	auto first = Mock::make();
	auto second = Mock::make();
	allocateMany(first, 15000);
	allocateMany(second, 15000);
	require(Mock::created == 30, "grow beyond fixed global pool");
	for (uint32_t frame = 0; frame < 12; ++frame)
	{
		check(first.reset());
		allocateMany(first, 15000);
	}
	require(Mock::created == 30, "reuse pools without continued growth");
	require(Mock::resetCount == 180, "reset only selected frame slot");
	first.shutdown();
	second.shutdown();
	require(Mock::pools.empty(), "shutdown releases every pool");
	std::puts("PASS growth, reuse, frame isolation and shutdown");

	for (VkResult result : {VK_ERROR_OUT_OF_POOL_MEMORY, VK_ERROR_FRAGMENTED_POOL,
		VK_ERROR_OUT_OF_HOST_MEMORY, VK_ERROR_OUT_OF_DEVICE_MEMORY, VK_ERROR_DEVICE_LOST})
	{
		auto allocator = Mock::make();
		allocateMany(allocator, 1);
		const auto before = Mock::created;
		Mock::allocateResult = result;
		VkDescriptorSet set = (VkDescriptorSet)(uintptr_t)1;
		require(allocator.allocate(VK_NULL_HANDLE, &set) == result, "preserve allocation error");
		require(set == VK_NULL_HANDLE, "never return stale handle on error");
		const bool recoverable = result == VK_ERROR_OUT_OF_POOL_MEMORY || result == VK_ERROR_FRAGMENTED_POOL;
		require(Mock::created == before + (recoverable ? 1 : 0), "bounded retry on pool errors only");
		Mock::allocateResult = VK_SUCCESS;
		allocator.shutdown();
	}
	std::puts("PASS pool exhaustion, fragmentation and terminal errors");
	for (VkResult result : {VK_ERROR_OUT_OF_POOL_MEMORY, VK_ERROR_FRAGMENTED_POOL})
	{
		auto allocator = Mock::make();
		allocateMany(allocator, 1);
		Mock::allocateResult = result;
		Mock::failOnce = true;
		allocateMany(allocator, 1);
		require(Mock::pools.size() == 2, "recover allocation in a fresh pool");
		Mock::failOnce = false;
		allocator.shutdown();
	}

	auto allocator = Mock::make();
	Mock::capacity = 7;
	allocateMany(allocator, 22);
	require(Mock::pools.size() == 4, "recover early pool exhaustion");
	Mock::capacity = 1024;
	Mock::resetResult = VK_ERROR_DEVICE_LOST;
	require(allocator.reset() == VK_ERROR_DEVICE_LOST, "propagate reset error");
	Mock::resetResult = VK_SUCCESS;
	check(allocator.reset());
	allocateMany(allocator, 1);
	allocator.shutdown();
	Mock::createResult = VK_ERROR_OUT_OF_DEVICE_MEMORY;
	VkDescriptorSet set = (VkDescriptorSet)(uintptr_t)1;
	require(allocator.allocate(VK_NULL_HANDLE, &set) == Mock::createResult && set == VK_NULL_HANDLE,
		"handle pool creation failure");
	Mock::createResult = VK_SUCCESS;
	allocator.shutdown();
	require(Mock::pools.empty(), "error paths leak no pools");
	std::puts("PASS creation/reset failure and early pool rollover");
}

static uint32_t validationErrors = 0;
static uint32_t realPoolsCreated = 0;

static VKAPI_ATTR VkBool32 VKAPI_CALL validation(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
	VkDebugUtilsMessageTypeFlagsEXT, const VkDebugUtilsMessengerCallbackDataEXT* data, void*)
{
	if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
	{
		++validationErrors;
		std::fprintf(stderr, "%s\n", data->pMessage);
	}
	return VK_FALSE;
}

static VKAPI_ATTR VkResult VKAPI_CALL createRealPool(VkDevice device, const VkDescriptorPoolCreateInfo* info,
	const VkAllocationCallbacks* callbacks, VkDescriptorPool* pool)
{
	const auto result = vkCreateDescriptorPool(device, info, callbacks, pool);
	if (result == VK_SUCCESS) ++realPoolsCreated;
	return result;
}

static void vulkanTest(bool legacy)
{
	uint32_t layerCount = 0;
	check(vkEnumerateInstanceLayerProperties(&layerCount, nullptr));
	std::vector<VkLayerProperties> layers(layerCount);
	check(vkEnumerateInstanceLayerProperties(&layerCount, layers.data()));
	bool validationAvailable = false;
	for (const auto& layer : layers)
		validationAvailable |= std::strcmp(layer.layerName, "VK_LAYER_KHRONOS_validation") == 0;
	require(validationAvailable, "Khronos validation layer is required");
	const char* layerName = "VK_LAYER_KHRONOS_validation";
	const char* extension = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
	VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
	app.apiVersion = VK_API_VERSION_1_1;
	VkInstanceCreateInfo instanceInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
	instanceInfo.pApplicationInfo = &app;
	instanceInfo.enabledLayerCount = 1;
	instanceInfo.ppEnabledLayerNames = &layerName;
	instanceInfo.enabledExtensionCount = 1;
	instanceInfo.ppEnabledExtensionNames = &extension;
	VkInstance instance;
	check(vkCreateInstance(&instanceInfo, nullptr, &instance));
	const auto createMessenger = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
	const auto destroyMessenger = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
	VkDebugUtilsMessengerCreateInfoEXT messengerInfo{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
	messengerInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
	messengerInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
	messengerInfo.pfnUserCallback = validation;
	VkDebugUtilsMessengerEXT messenger;
	check(createMessenger(instance, &messengerInfo, nullptr, &messenger));
	uint32_t gpuCount = 0;
	check(vkEnumeratePhysicalDevices(instance, &gpuCount, nullptr));
	require(gpuCount != 0, "Vulkan GPU available");
	std::vector<VkPhysicalDevice> gpus(gpuCount);
	check(vkEnumeratePhysicalDevices(instance, &gpuCount, gpus.data()));
	VkPhysicalDevice gpu = gpus[0];
	VkPhysicalDeviceProperties properties;
	vkGetPhysicalDeviceProperties(gpu, &properties);
	std::printf("GPU: %s\n", properties.deviceName);
	uint32_t familyCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(gpu, &familyCount, nullptr);
	std::vector<VkQueueFamilyProperties> families(familyCount);
	vkGetPhysicalDeviceQueueFamilyProperties(gpu, &familyCount, families.data());
	uint32_t family = 0;
	while (family < familyCount && !(families[family].queueFlags & VK_QUEUE_COMPUTE_BIT)) ++family;
	require(family < familyCount, "compute queue available");
	float priority = 1;
	VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
	queueInfo.queueFamilyIndex = family;
	queueInfo.queueCount = 1;
	queueInfo.pQueuePriorities = &priority;
	VkDeviceCreateInfo deviceInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
	deviceInfo.queueCreateInfoCount = 1;
	deviceInfo.pQueueCreateInfos = &queueInfo;
	VkDevice device;
	check(vkCreateDevice(gpu, &deviceInfo, nullptr, &device));
	VkQueue queue;
	vkGetDeviceQueue(device, family, 0, &queue);
	VkDescriptorSetLayoutBinding binding{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
	VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
	layoutInfo.bindingCount = 1;
	layoutInfo.pBindings = &binding;
	VkDescriptorSetLayout layout;
	check(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &layout));
	if (legacy)
	{
		const VkDescriptorPoolSize sizes[] = {
			{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 3072 * 16},
			{VK_DESCRIPTOR_TYPE_SAMPLER, 3072 * 16},
			{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 3072 * 2},
			{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3072 * 16},
			{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3072 * 16}
		};
		VkDescriptorPoolCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
		info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
		info.maxSets = 3072;
		info.poolSizeCount = 5;
		info.pPoolSizes = sizes;
		VkDescriptorPool pool;
		check(vkCreateDescriptorPool(device, &info, nullptr, &pool));
		VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
		allocation.descriptorPool = pool;
		allocation.descriptorSetCount = 1;
		allocation.pSetLayouts = &layout;
		uint32_t allocated = 0;
		VkResult result = VK_SUCCESS;
		VkDescriptorSet set = VK_NULL_HANDLE;
		while (allocated < 65536)
		{
			result = vkAllocateDescriptorSets(device, &allocation, &set);
			if (result != VK_SUCCESS) break;
			++allocated;
		}
		std::printf("Legacy pool: %u successful allocations, VkResult %d, null set %d\n",
			allocated, result, set == VK_NULL_HANDLE);
		require(result == VK_ERROR_OUT_OF_POOL_MEMORY && set == VK_NULL_HANDLE,
			"reproduce unchecked allocation failure before descriptor update");
		vkDestroyDescriptorPool(device, pool, nullptr);
		vkDestroyDescriptorSetLayout(device, layout, nullptr);
		vkDestroyDevice(device, nullptr);
		destroyMessenger(instance, messenger, nullptr);
		vkDestroyInstance(instance, nullptr);
		require(validationErrors == 0, "legacy failure reproduction uses valid Vulkan calls");
		return;
	}
	VkPipelineLayoutCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
	pipelineInfo.setLayoutCount = 1;
	pipelineInfo.pSetLayouts = &layout;
	VkPipelineLayout pipelineLayout;
	check(vkCreatePipelineLayout(device, &pipelineInfo, nullptr, &pipelineLayout));
	VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	bufferInfo.size = 64;
	bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
	VkBuffer buffer;
	check(vkCreateBuffer(device, &bufferInfo, nullptr, &buffer));
	VkMemoryRequirements requirements;
	vkGetBufferMemoryRequirements(device, buffer, &requirements);
	uint32_t memoryType = 0;
	while (!(requirements.memoryTypeBits & (1u << memoryType))) ++memoryType;
	VkMemoryAllocateInfo memoryInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
	memoryInfo.allocationSize = requirements.size;
	memoryInfo.memoryTypeIndex = memoryType;
	VkDeviceMemory memory;
	check(vkAllocateMemory(device, &memoryInfo, nullptr, &memory));
	check(vkBindBufferMemory(device, buffer, memory, 0));
	struct Slot
	{
		Allocator descriptors;
		VkCommandPool pool;
		VkCommandBuffer commands;
		VkFence fence;
	} slots[3];
	for (auto& slot : slots)
	{
		slot.descriptors.init({device, nullptr, createRealPool, vkDestroyDescriptorPool,
			vkResetDescriptorPool, vkAllocateDescriptorSets, 16});
		VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
		poolInfo.queueFamilyIndex = family;
		check(vkCreateCommandPool(device, &poolInfo, nullptr, &slot.pool));
		VkCommandBufferAllocateInfo allocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
		allocInfo.commandPool = slot.pool;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandBufferCount = 1;
		check(vkAllocateCommandBuffers(device, &allocInfo, &slot.commands));
		VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
		fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
		check(vkCreateFence(device, &fenceInfo, nullptr, &slot.fence));
	}
	for (uint32_t frame = 0; frame < 12; ++frame)
	{
		auto& slot = slots[frame % 3];
		check(vkWaitForFences(device, 1, &slot.fence, VK_TRUE, UINT64_MAX));
		check(vkResetCommandPool(device, slot.pool, 0));
		check(slot.descriptors.reset());
		VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
		check(vkBeginCommandBuffer(slot.commands, &begin));
		for (uint32_t i = 0; i < 6000; ++i)
		{
			VkDescriptorSet set;
			check(slot.descriptors.allocate(layout, &set));
			VkDescriptorBufferInfo descriptorBuffer{buffer, 0, 64};
			VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
			write.dstSet = set;
			write.descriptorCount = 1;
			write.descriptorType = binding.descriptorType;
			write.pBufferInfo = &descriptorBuffer;
			vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
			uint32_t offset = 0;
			vkCmdBindDescriptorSets(slot.commands, VK_PIPELINE_BIND_POINT_COMPUTE,
				pipelineLayout, 0, 1, &set, 1, &offset);
		}
		check(vkEndCommandBuffer(slot.commands));
		check(vkResetFences(device, 1, &slot.fence));
		VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
		submit.commandBufferCount = 1;
		submit.pCommandBuffers = &slot.commands;
		check(vkQueueSubmit(queue, 1, &submit, slot.fence));
	}
	check(vkDeviceWaitIdle(device));
	require(realPoolsCreated == 18, "real pools remain stable after warmup");
	for (auto& slot : slots)
	{
		vkDestroyCommandPool(device, slot.pool, nullptr);
		slot.descriptors.shutdown();
		vkDestroyFence(device, slot.fence, nullptr);
	}
	vkDestroyBuffer(device, buffer, nullptr);
	vkFreeMemory(device, memory, nullptr);
	vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
	vkDestroyDescriptorSetLayout(device, layout, nullptr);
	vkDestroyDevice(device, nullptr);
	destroyMessenger(instance, messenger, nullptr);
	vkDestroyInstance(instance, nullptr);
	require(validationErrors == 0, "zero Vulkan validation errors");
	std::puts("PASS 72000 descriptor allocations/updates/binds, 3 GPU slots, 18 reused pools, zero validation errors");
}

int main(int argc, char** argv)
{
	if (argc == 2 && std::strcmp(argv[1], "--vulkan") == 0) vulkanTest(false);
	else if (argc == 2 && std::strcmp(argv[1], "--legacy") == 0) vulkanTest(true);
	else unitTests();
}
