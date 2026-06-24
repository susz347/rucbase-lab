# NSG 查询性能优化 — 实验源码

基于 [NSG (Navigating Spreading-out Graph)](https://github.com/ZJULearning/nsg) 的修改版本，针对**方向A：查询性能优化**，实现了三项核心优化。

## 优化内容

| 优化项 | 文件 | 说明 |
|--------|------|------|
| Version Array | `include/efanna2e/index_nsg_optimized.h`, `src/index_nsg_optimized.cpp` | 用全局版本计数器替代每查询 O(n) 的 flags 重置 |
| FMA SIMD | `include/efanna2e/distance.h` | AVX2 FMA 指令替换分离的乘法和加法 |
| 增强预取 | `src/index_nsg_optimized.cpp` | T0/T1 两级预取 hint，优化 cache 层级利用 |

基线版本：`tests/test_nsg_optimized_search.cpp`（使用 `test_nsg_optimized_search.exe`）
优化版本：`tests/test_nsg_optimized_v2.cpp`（使用 `test_nsg_optimized_v2.exe`）

## 依赖

### Windows (MSVC)
- Visual Studio 2022+ （含 MSVC v14.50+）
- Windows SDK 10+
- Python 3.10+（用于构建脚本和基准测试）
- CPU 需支持 AVX2 + FMA 指令集

### Linux (GCC)
- GCC 4.9+ 并支持 OpenMP
- CMake 2.8+
- CPU 需支持 AVX2 + FMA 指令集

## 编译

### Windows (MSVC)

```bash
# 在项目根目录（nsg/）下执行：
python build_msvc.py build      # 编译
python build_msvc.py rebuild    # 清理后重新编译
python build_msvc.py clean      # 清理编译产物
```

编译产物输出到 `nsg/build_msvc/`：
- `test_nsg_index.exe` — 索引构建
- `test_nsg_search.exe` — 标准搜索（内存友好）
- `test_nsg_optimized_search.exe` — **基线**搜索（使用原始 flags）
- `test_nsg_optimized_v2.exe` — **优化版**搜索（Version Array + FMA + 预取）

### Linux (GCC)

```bash
cd nsg/
mkdir build && cd build/
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
```

## 运行

### 步骤 1：准备数据

数据文件需满足 `fvecs/ivecs` 格式。可使用项目根目录下的脚本生成合成数据：

```bash
cd ..
python generate_data.py
```

这会生成以下文件到 `data/` 目录：
- `sift_base.fvecs` — 基础向量
- `sift_query.fvecs` — 查询向量
- `sift_200nn.graph` — kNN 图
- `sift_groundtruth.ivecs` — groundtruth

### 步骤 2：构建 NSG 索引

```bash
cd nsg/build_msvc/
./test_nsg_index.exe ../../data/sift_base.fvecs ../../data/sift_200nn.graph 40 50 500 ../../data/sift.nsg
```

参数：`DATA_PATH KNNG_PATH L R C NSG_PATH`
- `L=40`：NSG 质量参数
- `R=50`：最大出度
- `C=500`：构建时候选池大小

### 步骤 3：运行搜索

**基线版本**（原始 flags 方案）：
```bash
./test_nsg_optimized_search.exe ../../data/sift_base.fvecs ../../data/sift_query.fvecs ../../data/sift.nsg 120 100 result_base.ivecs
```

**优化版本**（Version Array + FMA + 预取）：
```bash
./test_nsg_optimized_v2.exe ../../data/sift_base.fvecs ../../data/sift_query.fvecs ../../data/sift.nsg 120 100 result_opt.ivecs
```

参数：`DATA_PATH QUERY_PATH NSG_PATH SEARCH_L SEARCH_K RESULT_PATH`

### 步骤 4：运行基准测试

在项目根目录下执行：

```bash
python reproduce.py          # 一键复现（生成数据→构建索引→运行A/B对比）
python reproduce.py --data-only   # 仅生成数据
python reproduce.py --benchmark   # 仅运行基准测试
```

## 目录结构

```
nsg/
  include/efanna2e/
    distance.h                   -- 距离计算（FMA 优化）
    index.h                      -- 基类接口
    index_nsg.h                  -- NSG 索引类
    index_nsg_optimized.h        -- 优化版索引（Version Array）
    neighbor.h                   -- 邻居结构
    parameters.h                 -- 参数传递
    util.h                       -- 数据对齐工具（MSVC 适配）
  src/
    index.cpp                    -- Index 基类实现
    index_nsg.cpp                -- NSG 核心（OpenMP MSVC 兼容修复）
    index_nsg_optimized.cpp      -- 优化版搜索实现
  tests/
    test_nsg_index.cpp           -- 索引构建
    test_nsg_search.cpp          -- 标准搜索
    test_nsg_optimized_search.cpp    -- 基线搜索测试
    test_nsg_optimized_v2.cpp        -- 优化版搜索测试
  build_msvc.py                  -- MSVC 构建脚本
  CMakeLists.txt                 -- CMake 构建（Linux）
  README.md                      -- 本文件
```

## 许可

原始 NSG 采用 MIT 许可证。本修改版本用于课程实验。
