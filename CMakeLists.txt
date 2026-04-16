cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(kerrtrace VERSION 1.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── LibTorch ──────────────────────────────────────────────────────────────────
# Set via: cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# ── FetchContent deps ─────────────────────────────────────────────────────────
include(FetchContent)

FetchContent_Declare(cli11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG        v2.4.1
    GIT_SHALLOW    TRUE)

FetchContent_Declare(nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE)

FetchContent_Declare(stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE)

FetchContent_MakeAvailable(cli11 nlohmann_json stb)

# ── OpenMP (optional, used for CPU tile parallelism) ─────────────────────────
find_package(OpenMP QUIET)

# ── Sources ───────────────────────────────────────────────────────────────────
add_executable(kerrtrace
    src/main.cpp
    src/config.cpp
    src/geometry.cpp
    src/palette.cpp
    src/raytracer.cpp
    src/animation.cpp
    src/image_io.cpp
)

target_include_directories(kerrtrace PRIVATE
    include
    ${stb_SOURCE_DIR}
)

target_link_libraries(kerrtrace PRIVATE
    "${TORCH_LIBRARIES}"
    CLI11::CLI11
    nlohmann_json::nlohmann_json
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(kerrtrace PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(kerrtrace PRIVATE KT_USE_OPENMP)
endif()

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS kerrtrace DESTINATION bin)

# ── Platform-specific RPATH fix for LibTorch shared libs ─────────────────────
if(APPLE)
    set_target_properties(kerrtrace PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "@loader_path/../lib"
    )
elseif(UNIX)
    set_target_properties(kerrtrace PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "$ORIGIN/../lib"
    )
endif()
