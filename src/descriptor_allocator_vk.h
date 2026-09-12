#ifndef BGFX_DESCRIPTOR_ALLOCATOR_VK_H_HEADER_GUARD
#define BGFX_DESCRIPTOR_ALLOCATOR_VK_H_HEADER_GUARD

namespace bgfx { namespace vk
{
	/// Owns descriptor pools for one command-buffer slot. Reset only after its fence completes.
	class DescriptorAllocatorVK
	{
	public:
		struct Api
		{
			::VkDevice device;
			const ::VkAllocationCallbacks* allocator;
			PFN_vkCreateDescriptorPool createPool;
			PFN_vkDestroyDescriptorPool destroyPool;
			PFN_vkResetDescriptorPool resetPool;
			PFN_vkAllocateDescriptorSets allocateSets;
			uint32_t maxSamplers;
		};

		void init(const Api& _api)
		{
			m_api = _api;
		}

		VkResult allocate(::VkDescriptorSetLayout _layout, ::VkDescriptorSet* _set)
		{
			*_set = VK_NULL_HANDLE;
			for (;;)
			{
				if (m_currentPool == m_pools.size() )
				{
					const VkDescriptorPoolSize sizes[] =
					{
						{ VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,          kSetsPerPool * m_api.maxSamplers },
						{ VK_DESCRIPTOR_TYPE_SAMPLER,                kSetsPerPool * m_api.maxSamplers },
						{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, kSetsPerPool * 2                 },
						{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         kSetsPerPool * m_api.maxSamplers },
						{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          kSetsPerPool * m_api.maxSamplers },
					};
					VkDescriptorPoolCreateInfo info = {};
					info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
					info.maxSets = kSetsPerPool;
					info.poolSizeCount = uint32_t(sizeof(sizes) / sizeof(sizes[0]) );
					info.pPoolSizes = sizes;
					Pool pool;
					const VkResult result = m_api.createPool(m_api.device, &info, m_api.allocator, &pool.handle);
					if (VK_SUCCESS != result)
					{
						return result;
					}
					m_pools.push_back(pool);
				}

				Pool& pool = m_pools[m_currentPool];
				if (pool.used == kSetsPerPool)
				{
					++m_currentPool;
					continue;
				}

				VkDescriptorSetAllocateInfo info = {};
				info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
				info.descriptorPool = pool.handle;
				info.descriptorSetCount = 1;
				info.pSetLayouts = &_layout;
				const VkResult result = m_api.allocateSets(m_api.device, &info, _set);
				if (VK_SUCCESS == result)
				{
					++pool.used;
					return result;
				}

				*_set = VK_NULL_HANDLE;
				if (pool.used == 0
				||  (VK_ERROR_OUT_OF_POOL_MEMORY != result && VK_ERROR_FRAGMENTED_POOL != result) )
				{
					return result;
				}
				++m_currentPool;
			}
		}

		VkResult reset()
		{
			for (Pool& pool : m_pools)
			{
				const VkResult result = m_api.resetPool(m_api.device, pool.handle, 0);
				if (VK_SUCCESS != result)
				{
					return result;
				}
				pool.used = 0;
			}
			m_currentPool = 0;
			return VK_SUCCESS;
		}

		void shutdown()
		{
			for (const Pool& pool : m_pools)
			{
				m_api.destroyPool(m_api.device, pool.handle, m_api.allocator);
			}
			m_pools.clear();
			m_currentPool = 0;
		}

	private:
		static constexpr uint32_t kSetsPerPool = 1024;
		struct Pool
		{
			::VkDescriptorPool handle = VK_NULL_HANDLE;
			uint32_t used = 0;
		};
		Api m_api = {};
		stl::vector<Pool> m_pools;
		uint32_t m_currentPool = 0;
	};

} }

#endif
