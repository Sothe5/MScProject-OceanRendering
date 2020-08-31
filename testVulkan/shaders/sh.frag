#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 1) uniform samplerCube Cubemap;

layout(binding = 4) uniform sampler2D texSamplerCompute;

layout(location = 0) in vec3 fragColor;	// get all the values from the vertex shader in the fragment
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec4 fragNormalCoord;

layout(location = 4) in vec3 fragAmbientColor;
layout(location = 5) in vec3 fragDiffuseColor;
layout(location = 9) in vec4 fragLightVector;
layout(location = 10) in vec3 fragVectorFromCameraToPixel;
layout(location = 11) in vec3 fragNormal;
layout(location = 12) in vec3 fragLightPos;
layout(location = 14) in vec3 fragWorldPixelToCamera;

layout(location = 0) out vec4 outColor;

vec4 normLightVector = normalize(fragLightVector);
vec4 normNormal = normalize(fragNormalCoord);

vec3 computeAmbientLight()
{
	return  fragAmbientColor*0.2f;	// compute ambient light
}

vec3 computeDiffuseLight()
{
	float diffuseDot = dot(normLightVector, normNormal);
	return fragDiffuseColor  * diffuseDot;	// compute diffuse light 
}

void main() {
	vec3 ambient = computeAmbientLight();
	vec3 diffuse = computeDiffuseLight();

	vec3 lightResult = ambient + diffuse;
	
	vec3 reflect_vector = reflect( fragVectorFromCameraToPixel, fragNormal );
	vec4 reflect_color = texture( Cubemap, reflect_vector );

	vec3 H = normalize(fragLightPos + fragWorldPixelToCamera);

	float fresnel = 0.02 + 0.98 * pow(1.0 - abs(dot(fragNormal, fragVectorFromCameraToPixel)), 5.0);

	outColor = vec4(lightResult * mix(fragColor, reflect_color.xyz, fresnel),1);
}