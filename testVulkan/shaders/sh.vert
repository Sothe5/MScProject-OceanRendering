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

layout( push_constant ) uniform pushConstants {
  int isWavy;
} pc;

layout(binding = 18) uniform sampler2D texSamplerHeightMap;
layout(binding = 37) uniform sampler2D texSamplerSlopeX;
layout(binding = 40) uniform sampler2D texSamplerSlopeZ;
layout(binding = 43) uniform sampler2D texSamplerDispX;
layout(binding = 46) uniform sampler2D texSamplerDispZ;

layout(location = 0) in vec3 inPosition;	// position
layout(location = 1) in vec3 inColor;	// color
layout(location = 2) in vec2 inTexCoord;	// textCoord
layout(location = 3) in vec3 inNormalCoord;	// normal



layout(location = 0) out vec3 fragColor;	// variables that will be passed to the fragment shader
layout(location = 1) out vec2 fragTexCoord;
layout(location = 2) out vec4 fragNormalCoord;

layout(location = 4) out vec3 fragAmbientColor;
layout(location = 5) out vec3 fragDiffuseColor;
layout(location = 9) out vec4 fragLightVector;
layout(location = 10) out vec3 fragVectorFromCameraToPixel;
layout(location = 11) out vec3 fragNormal;
layout(location = 12) out vec3 fragLightPos;
layout(location = 14) out vec3 fragWorldPixelToCamera;

void main() {

	vec4 textureSampledHeightMap = texture(texSamplerHeightMap, inTexCoord);
	vec4 textureSampledSlopeX = texture(texSamplerSlopeX, inTexCoord);
	vec4 textureSampledSlopeZ = texture(texSamplerSlopeZ, inTexCoord);
	vec4 textureSampledDispX = texture(texSamplerDispX, inTexCoord);
	vec4 textureSampledDispZ = texture(texSamplerDispZ, inTexCoord);
	
	vec4 WCS_position;
	if(pc.isWavy == 1)
	{
		WCS_position = ubo.model * vec4(vec3(inPosition.x, inPosition.y + textureSampledHeightMap.x, inPosition.z ), 1.0);	// Position in VCS
	}
	else
	{
		WCS_position = ubo.model * vec4(vec3(inPosition.x + textureSampledDispX.x , inPosition.y + textureSampledHeightMap.x, inPosition.z + textureSampledDispZ.x ), 1.0);
	}

	vec4 VCS_position = ubo.view * WCS_position;	// Position in VCS
    gl_Position = ubo.proj * VCS_position;	// position in NVCS
    fragColor = inColor;	// base color of the triangle
    fragTexCoord = inTexCoord;	// texture coordinates

	vec3 normalCoords = normalize(vec3(textureSampledSlopeX.x, 1.0f, textureSampledSlopeZ.x));	// normalize normal coordinates

	fragNormalCoord = ubo.view * ubo.model * vec4(normalCoords,0.0);	// normal in VCS
	fragLightVector =  ubo.view * ubl.lightPosition - VCS_position;	// vector from the pixel to the light
	fragAmbientColor = ubl.ambientColor;	// ambient light parameters
	fragDiffuseColor = ubl.diffuseColor;	// diffuse light parameters
	fragVectorFromCameraToPixel = normalize(inPosition - ubo.cameraPosition); // vector from the camera to the pixel in WCS
	fragNormal = normalCoords;	// normal coordinates in WCS
	fragLightPos = ubl.lightPosition.xyz;	// light position in WCS
	fragWorldPixelToCamera = normalize(ubo.cameraPosition - inPosition);	// vector from the pixel to the camera in WCS
}