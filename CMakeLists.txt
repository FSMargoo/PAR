cmake_minimum_required(VERSION 4.0)
project(PAR)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(./thirdparty/glfw)
add_subdirectory(./thirdparty/glm)

set(IMGUICode thirdparty/imgui/imgui.cpp thirdparty/imgui/imgui_draw.cpp thirdparty/imgui/imgui_widgets.cpp
        thirdparty/imgui/backends/imgui_impl_opengl3.cpp thirdparty/imgui/backends/imgui_impl_glfw.cpp
        thirdparty/imgui/imgui_demo.cpp thirdparty/imgui/imgui_draw.cpp thirdparty/imgui/imgui_tables.cpp)

add_library(IMGUI ${IMGUICode})
target_link_libraries(IMGUI glfw opengl32)

add_executable(PAR_Scene1 scene1.cpp thirdparty/glad/src/glad.c)
add_executable(PAR_Scene2 scene2.cpp thirdparty/glad/src/glad.c)
add_executable(PAR_Scene3 scene3.cpp thirdparty/glad/src/glad.c)
add_executable(PAR_Scene4 scene4.cpp thirdparty/glad/src/glad.c)

target_include_directories(PAR_Scene1 PUBLIC ./thirdparty/glad/include)
target_include_directories(PAR_Scene1 PUBLIC ./thirdparty/imgui)
target_include_directories(PAR_Scene1 PUBLIC ./thirdparty/glm)

target_include_directories(PAR_Scene2 PUBLIC ./thirdparty/glad/include)
target_include_directories(PAR_Scene2 PUBLIC ./thirdparty/imgui)
target_include_directories(PAR_Scene2 PUBLIC ./thirdparty/glm)

target_include_directories(PAR_Scene3 PUBLIC ./thirdparty/glad/include)
target_include_directories(PAR_Scene3 PUBLIC ./thirdparty/imgui)
target_include_directories(PAR_Scene3 PUBLIC ./thirdparty/glm)

target_include_directories(PAR_Scene4 PUBLIC ./thirdparty/glad/include)
target_include_directories(PAR_Scene4 PUBLIC ./thirdparty/imgui)
target_include_directories(PAR_Scene4 PUBLIC ./thirdparty/glm)

target_include_directories(IMGUI PUBLIC ./thirdparty/imgui)

target_link_libraries(PAR_Scene1 glfw opengl32 IMGUI glm)
target_link_libraries(PAR_Scene2 glfw opengl32 IMGUI glm)
target_link_libraries(PAR_Scene3 glfw opengl32 IMGUI glm)
target_link_libraries(PAR_Scene4 glfw opengl32 IMGUI glm)
