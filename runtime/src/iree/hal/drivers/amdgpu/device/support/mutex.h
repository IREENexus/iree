// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// NOTE: these structs/enums are taken from the HSA spec, the hsa.h and
// hsa_ext_amd.h headers, and the LLVM AMDGPU device library headers.
// We define them locally as the HSA headers cannot be directly used in OpenCL
// and the device libraries are only available in a fork of LLM.
//
// Sources:
// https://hsafoundation.com/wp-content/uploads/2021/02/HSA-SysArch-1.2.pdf
// https://github.com/ROCm/ROCR-Runtime
// https://github.com/ROCm/rocMLIR/blob/develop/external/llvm-project/amd/device-libs/README.md

#ifndef IREE_HAL_DRIVERS_AMDGPU_DEVICE_SUPPORT_MUTEX_H_
#define IREE_HAL_DRIVERS_AMDGPU_DEVICE_SUPPORT_MUTEX_H_

#include "iree/hal/drivers/amdgpu/device/support/opencl.h"

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_device_mutex_t
//===----------------------------------------------------------------------===//

// Device spin-lock mutex.
// This can run on the host as well but is optimized for device usage. Spinning
// on the host is a bad idea.
//
// https://rigtorp.se/spinlock/
typedef iree_hal_amdgpu_device_atomic_uint32_t iree_hal_amdgpu_device_mutex_t;

#define IREE_HAL_AMDGPU_DEVICE_MUTEX_UNLOCKED 0u
#define IREE_HAL_AMDGPU_DEVICE_MUTEX_LOCKED 1u

// Initializes a mutex to the unlocked state.
static inline void iree_hal_amdgpu_device_mutex_initialize(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_mutex_t* IREE_OCL_RESTRICT
        out_mutex) {
  IREE_HAL_AMDGPU_DEVICE_ATOMIC_INIT(out_mutex,
                                     IREE_HAL_AMDGPU_DEVICE_MUTEX_UNLOCKED);
}

// Spins until a lock on the mutex is acquired.
static inline void iree_hal_amdgpu_device_mutex_lock(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_mutex_t* IREE_OCL_RESTRICT mutex) {
  for (;;) {
    // Optimistically assume the lock is free on the first try.
    uint32_t prev = IREE_HAL_AMDGPU_DEVICE_MUTEX_UNLOCKED;
    if (iree_hal_amdgpu_device_atomic_compare_exchange_strong_explicit(
            mutex, &prev, IREE_HAL_AMDGPU_DEVICE_MUTEX_LOCKED,
            iree_hal_amdgpu_device_memory_order_acquire,
            iree_hal_amdgpu_device_memory_order_acquire,
            iree_hal_amdgpu_device_memory_scope_all_svm_devices)) {
      return;
    }
    // Wait for lock to be released without generating cache misses.
    while (iree_hal_amdgpu_device_atomic_load_explicit(
        mutex, iree_hal_amdgpu_device_memory_order_relaxed,
        iree_hal_amdgpu_device_memory_scope_all_svm_devices)) {
      // Yield for a bit to give the other thread a chance to unlock.
      iree_hal_amdgpu_device_yield();
    }
  }
}

// Unlocks a mutex. Must be called with the lock held by the caller.
static inline void iree_hal_amdgpu_device_mutex_unlock(
    IREE_OCL_GLOBAL iree_hal_amdgpu_device_mutex_t* IREE_OCL_RESTRICT mutex) {
  iree_hal_amdgpu_device_atomic_store_explicit(
      mutex, IREE_HAL_AMDGPU_DEVICE_MUTEX_UNLOCKED,
      iree_hal_amdgpu_device_memory_order_release,
      iree_hal_amdgpu_device_memory_scope_all_svm_devices);
}

#endif  // IREE_HAL_DRIVERS_AMDGPU_DEVICE_SUPPORT_MUTEX_H_
