cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(module_1D_2L_chart)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(module_1D_2L_chart module_1D_2L_chart.cpp)
target_link_libraries(module_1D_2L_chart "${TORCH_LIBRARIES}")
set_property(TARGET module_1D_2L_chart PROPERTY CXX_STANDARD 17)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET module_1D_2L_chart
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:module_1D_2L_chart>)
endif (MSVC)