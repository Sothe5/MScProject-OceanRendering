#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject {	// Model View Projection matrices
    mat4 model;
	mat4 modelCube;
    mat4 view;
    mat4 viewCube;
    mat4 proj;
	vec3 cameraPosition;
} ubo;
	
layout(binding = 2) uniform UniformBufferLight {	// values of the material related to lighting
    float specularHighlight;
	vec3 ambientColor;
	vec3 diffuseColor;
	vec3 specularColor;
	vec3 emissiveColor;
	vec4 lightPosition;
} ubl;

layout(location = 0) in vec3 inPosition;	// position

layout(location = 1) out vec3 vertTextCoord;

void main() {
	vec4 VCS_position = ubo.viewCube * ubo.modelCube * vec4(inPosition, 1.0);	// Position in VCS

    gl_Position = ubo.proj * VCS_position;	// position in NVCS
    vertTextCoord = inPosition;
}