#version 410

layout(location = 0) in vec2 IPosition;
layout(location = 1) in vec2 IUv;
out vec2 fragUV;

void main() {
    fragUV = IUv;

    gl_Position = vec4(IPosition, 0.0, 1.0);
}