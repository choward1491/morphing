# Implementation Plan: High-Performance Real-RAM Planar Graph Morphing Engine

## 1. Motivation & Design Philosophy
Theoretical computational geometry assumes a real-RAM model of computation where mathematical operations on real numbers are exact and execute in constant time. Translating this to finite-precision silicon typically introduces topological collapses, collinearity errors, and intersecting edges. 

This engine bridges that gap by isolating the exact, arbitrary-precision mathematical proofs to the CPU (using deterministic geometric predicates and integer/rational arithmetic) and completely decoupling them from the standard-precision visual interpolation handled by the GPU. By leveraging a flat, contiguous memory architecture and a data-oriented design, the system avoids the heap fragmentation that usually cripples arbitrary-precision geometry algorithms, achieving high throughput without sacrificing exactness.

## 2. Core Dependencies
To maintain a hermetic build and utilize enterprise-grade infrastructure, the project relies on the following ecosystem:

* **Bazel** (bazel.build): The primary build system, ensuring fast, reproducible builds and seamless integration of C++ proto libraries.
* **Abseil** (abseil.io): Core C++ utilities, specifically utilized for `absl::Status` and `absl::StatusOr` to cleanly handle deep topological validation errors without exception handling overhead.
* **Protocol Buffers** (protobuf.dev): Used for defining strict configuration schemas, input/target embeddings, morph sequence definitions, and animation output parameters.
* **Boost.Multiprecision** (boost.org): Provides the exact integer and rational number types needed to evaluate 4x4 determinant signs and cross-products without floating-point drift. 
* **Eigen** (eigen.tuxfamily.org): A high-performance linear algebra library for evaluating linear transformations, quaternions, and QR factorizations.
* **Metal-cpp** (developer.apple.com/metal/cpp/): Apple's native C++ interface for the Metal graphics API, used for hardware-accelerated interpolation of longitudinal morphs.
* **GLFW** (glfw.org): A lightweight, cross-platform library for managing the OS window and capturing mouse/keyboard inputs for the interactive visualizer.
* **FFmpeg** (ffmpeg.org): Invoked as a sub-process via POSIX pipes (`popen`) to encode raw offscreen pixel buffers into MP4 video files.
* **Skia** (skia.org) or **Core Graphics**: 2D vector graphics engines for rendering exact stylized representations (dashed lines, faded borders) of the spherical embeddings. Skia provides a direct Metal backend and SVG export, while Core Graphics is a lightweight native macOS alternative.

## 3. Architectural Pillars

### A. Array-Backed Half-Edge (DCEL) Structure
Instead of scattered pointer-based objects or hash maps, the graph topology is stored in pre-allocated, flat `std::vector` arrays. 
* **Vertices and Half-Edges:** Represented strictly by integer indices pointing into these arrays.
* **Active Masks:** `std::vector<uint8_t>` or `std::bitset` tracks the active state of vertices and faces. During an edge contraction, degenerate faces are simply masked as inactive rather than dynamically deleted, maintaining cache locality and O(1) topological updates.

### B. Data-Oriented Topological Events
The pseudomorph sequence is built inductively using a `std::variant` to represent a closed set of operations (e.g., `EdgeContraction`, `EdgeExpansion`, `VertexMove`).
* **Zero Vtables:** Storing these variants in a `std::deque` eliminates virtual function overhead and pointer chasing.
* **Reversibility:** Pure value semantics allow the sequence to be trivially reversed (`std::reverse`) to calculate the epsilon-perturbations for the morph expansion.

### C. Deterministic Constant-Time Linear Programming
Instead of heavy LP solvers (like SoPlex or Seidel's randomized algorithm), the engine hand-rolls a deterministic, dual-space boundary evaluation.
* **The Dual Cone:** To find a hemisphere containing up to 5 vertices and a kernel point (N <= 6), the algorithm computes the cross-products of all pairs of vectors, generating at most 30 candidate extreme rays.
* **Exact Validation:** These rays are checked against the input vectors using exact dot products. The sum of the valid rays geometrically guarantees an exact interior normal vector.

## 4. Implementation Phases

### Phase 1: Geometric Primitives & Topology Foundations
* Implement the signed homogeneous coordinate structs (`[x, y, z, w]`).
* Build the flat Half-Edge arrays and the active/inactive masking logic.
* Write the exact determinant evaluation functions templated on Boost.Multiprecision types.

### Phase 2: The Pseudomorph Builder
* Implement the inductive edge-contraction logic.
* Integrate the deterministic dual-cone LP algorithm to validate kernel constraints.
* Push integer-based topological mutations into the `std::deque<std::variant<...>>`.

### Phase 3: The Morph Compiler
* Reverse the pseudomorph queue.
* Calculate the exact linear transformations and strict interior epsilon-perturbations for expanding vertices.
* Downcast the exact coordinates to 32-bit `float` and flatten the data into a contiguous `std::vector<MorphKeyframe>` for rendering ingestion.

### Phase 4: Visualization & Rendering
* Setup the GLFW window and bind a rendering layer.
* **3D Morph Compute:** Write a Metal Compute Shader to accept keyframes and a normalized time parameter `t` in [0, 1] to calculate parallel longitudinal shifts.
* **2D Stylized Rendering:** Utilize Skia (or Core Graphics) to calculate front/back view visibility using dot products against the camera vector. Draw dashed/faded primitives for the back hemisphere and solid/sharp primitives for the front.
* **Output Pipeline:** Build the headless offscreen texture pipeline that pipes raw RGB bytes directly to FFmpeg for animation, and leverage Skia's SVG canvas for static, publication-ready vector exports.

## 5. Potential Challenges to Investigate

* **Bit-Length Explosion in Exact Arithmetic:** When computing chained exact coordinates, the bit-lengths of rational numerators and denominators can grow exponentially, heavily bogging down CPU performance. Investigate when it is safe to strictly truncate or normalize these arbitrary-precision integers without violating planarity proofs.
* **Metal-cpp Memory Management:** Metal relies on `NS::AutoreleasePool`. When writing raw C++ loops to generate frames for FFmpeg, failing to properly scope your autorelease pools per-frame will result in massive memory leaks.
* **Data Layout for Compute Shaders:** Apple Silicon unifies CPU and GPU memory, meaning you can often pass pointers directly rather than explicitly copying buffers. Investigating `MTLStorageModeShared` vs. `MTLStorageModeManaged` will be critical for feeding your `MorphKeyframe` array to the shader with zero-copy overhead.