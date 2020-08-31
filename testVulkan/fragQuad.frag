#version 450
#extension GL_ARB_separate_shader_objects : enable


layout(binding = 5) uniform sampler2D texSamplerH0k;
layout(binding = 7) uniform sampler2D texSamplerH0kMinus;
layout(binding = 9) uniform sampler2D texSamplerHkt;
layout(binding = 12) uniform sampler2D texSamplerFFTAux;
layout(binding = 15) uniform sampler2D texSamplerAlternate1;
layout(binding = 17) uniform sampler2D texSamplerHeightMap;
layout(binding = 36) uniform sampler2D texSamplerSlopeX;
layout(binding = 39) uniform sampler2D texSamplerSlopeZ;
layout(binding = 42) uniform sampler2D texSamplerDispX;
layout(binding = 45) uniform sampler2D texSamplerDispZ;

layout(location = 1) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

layout( push_constant ) uniform pushConstants {
  int isH0k;
  int isH0kMinus;
  int isHkt;
  int isFFTAux;
  int isHorizontalFFT;
  int isVerticalFFT;
  int isHeightMap;
  int isSlopeX;
  int isSlopeZ;
  int isDispX;
  int isDispZ;
} pc;

void main() {

	if(pc.isH0k == 1)
	{
		vec4 textureSampled = texture(texSamplerH0k, fragTexCoord);
		outColor = vec4(textureSampled.xyz,1);
	}else if(pc.isH0kMinus == 1)
	{
		vec4 textureSampled = texture(texSamplerH0kMinus, fragTexCoord);
		outColor = vec4(textureSampled.xyz,1);
	}else if(pc.isHkt == 1)
	{
		vec4 textureSampled = texture(texSamplerHkt, fragTexCoord);
		outColor = vec4(textureSampled.xyz ,1);
	}else if(pc.isFFTAux == 1)
	{
		vec4 textureSampled = texture(texSamplerFFTAux, fragTexCoord);
		outColor = vec4(textureSampled.xyz,1);
	}else if(pc.isHorizontalFFT == 1)
	{
		vec4 textureSampled = texture(texSamplerHkt, fragTexCoord);
		outColor = vec4(textureSampled.xyz,1);
	}else if(pc.isVerticalFFT == 1)
	{
		vec4 textureSampled = texture(texSamplerAlternate1, fragTexCoord);
		outColor = vec4(textureSampled.xyz,1);
	}else if(pc.isHeightMap == 1)
	{
		vec4 textureSampled = texture(texSamplerHeightMap, fragTexCoord);
		outColor = vec4(textureSampled.xyz,1);
	}
	else if(pc.isSlopeX == 1)
	{
		vec4 textureSampled = texture(texSamplerSlopeX, fragTexCoord);
		outColor = vec4(textureSampled.xyz,1);
	}
	else if(pc.isSlopeZ == 1)
	{
		vec4 textureSampled = texture(texSamplerSlopeZ, fragTexCoord);
		outColor = vec4(textureSampled.xyz,1);
	}
	else if(pc.isDispX == 1)
	{
		vec4 textureSampled = texture(texSamplerDispX, fragTexCoord);
		outColor = vec4(textureSampled.xyz,1);
	}
	else if(pc.isDispZ == 1)
	{
		vec4 textureSampled = texture(texSamplerDispZ, fragTexCoord);
		outColor = vec4(textureSampled.xyz,1);
	}else
	{
		outColor = vec4(1);
	}
}