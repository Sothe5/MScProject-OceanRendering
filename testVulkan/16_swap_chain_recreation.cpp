#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>
#include <chrono>

#include <iostream> // for reporting and propagate errors
#include <fstream>
#include <stdexcept> // for reporting and propagate errors
#include <algorithm>
#include <vector>
#include <cstring>
#include <cstdlib> // provides the EXIT_SUCCESS and EXIT_FAILURE macros.
#include <cstdint>
#include <optional>
#include <set>
#include <array>
#include <fstream>
#include <string>
#include <sstream>
#include <unordered_map>
#include <random>

#include "imgui_impl_vulkan.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"

#include "Camera.h"

const int WIDTH = 1920;
const int HEIGHT = 1080;

const int MAX_FRAMES_IN_FLIGHT = 2;	// Number of simultaneus frames we allow

VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;

const std::vector<const char*> validationLayers = {
	"VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
	VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#ifdef NDEBUG
const bool enableValidationLayers = false;	// if in release mode, don't check for validation layers
#else
const bool enableValidationLayers = true;	// if in debug mode, check for validation layers
#endif

// function to create the VkDebugUtilsMessengerEXT object 
VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
	auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");	// look for the address and returns a nullptr if the function cannot be loaded
	if (func != nullptr) {
		return func(instance, pCreateInfo, pAllocator, pDebugMessenger);	// proxy function that handles the address issue
	}
	else {
		return VK_ERROR_EXTENSION_NOT_PRESENT;	// return and string to notify of an error with the extension
	}
}

// function to destroy the VkDebugUtilsMessengerEXT object 
void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
	auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"); // look for the address and returns a nullptr if the function cannot be loaded
	if (func != nullptr) {
		func(instance, debugMessenger, pAllocator);
	}
}

// It holds an index number for every queue in the system or nothing if it was not set up yet.
struct QueueFamilyIndices {
	std::optional<uint32_t> graphicsFamily;	// queue to support graphics commands
	std::optional<uint32_t> presentFamily;

	bool isComplete() {	// ask whether every queue has a index or not.
		return graphicsFamily.has_value() && presentFamily.has_value();
	}
};

struct SwapChainSupportDetails {
	VkSurfaceCapabilitiesKHR capabilities;	// number of images in swap change, width/height of images...
	std::vector<VkSurfaceFormatKHR> formats;	// pixel format, color space
	std::vector<VkPresentModeKHR> presentModes;	// the available presentation modes
};



// struct for all the information that every vertex will have
struct Vertex {
	glm::vec3 pos;
	glm::vec3 color;
	glm::vec2 texCoord;
	glm::vec3 normalCoord;

	static VkVertexInputBindingDescription getBindingDescription() {
		VkVertexInputBindingDescription bindingDescription = {};
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(Vertex);
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		return bindingDescription;
	}

	static std::array<VkVertexInputAttributeDescription, 4> getAttributeDescriptions() {
		std::array<VkVertexInputAttributeDescription, 4> attributeDescriptions = {};

		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(Vertex, pos);

		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(Vertex, color);

		attributeDescriptions[2].binding = 0;
		attributeDescriptions[2].location = 2;
		attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
		attributeDescriptions[2].offset = offsetof(Vertex, texCoord);

		attributeDescriptions[3].binding = 0;
		attributeDescriptions[3].location = 3;
		attributeDescriptions[3].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[3].offset = offsetof(Vertex, normalCoord);

		return attributeDescriptions;
	}

	bool operator==(const Vertex& other) const {	// overloaded operator of comparison in order to be able to selct which vertices are repeated
		return pos == other.pos && color == other.color && texCoord == other.texCoord && normalCoord == other.normalCoord;
	}
};

namespace std {
	template<> struct hash<Vertex> {	// hash function in order to be able to selct which vertices are repeated
		size_t operator()(Vertex const& vertex) const {
			return ((hash<glm::vec3>()(vertex.pos) ^ (hash<glm::vec3>()(vertex.color) << 1) ^ (hash<glm::vec2>()(vertex.texCoord) << 1)) >> 1) ^ (hash<glm::vec3>()(vertex.normalCoord) << 1);
		}
	};
}

struct VertexOnlyPos {
	glm::vec3 pos;

	static VkVertexInputBindingDescription getBindingDescription() {
		VkVertexInputBindingDescription bindingDescription = {};
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(VertexOnlyPos);
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		return bindingDescription;
	}

	static std::array<VkVertexInputAttributeDescription, 1> getAttributeDescriptions() {
		std::array<VkVertexInputAttributeDescription, 1> attributeDescriptions = {};

		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(VertexOnlyPos, pos);

		return attributeDescriptions;
	}

	bool operator==(const VertexOnlyPos& other) const {	// overloaded operator of comparison in order to be able to selct which vertices are repeated
		return pos == other.pos;
	}
};

namespace std {
	template<> struct hash<VertexOnlyPos> {	// hash function in order to be able to selct which vertices are repeated
		size_t operator()(VertexOnlyPos const& vertex) const {
			return hash<glm::vec3>()(vertex.pos);
		}
	};
}


struct UniformBufferObject {	// Union buffer with the information of the object
	alignas(16) glm::mat4 model;
	alignas(16) glm::mat4 modelCube;
	alignas(16) glm::mat4 view;
	alignas(16) glm::mat4 viewCube;
	alignas(16) glm::mat4 proj;
	alignas(16) glm::vec3 cameraPosition;
};

struct UniformBufferLight {	// Union buffer with the information of the light
	alignas(4) float specularHighlight;
	alignas(16) glm::vec3 ambientColor;
	alignas(16) glm::vec3 diffuseColor;
	alignas(16) glm::vec3 specularColor;
	alignas(16) glm::vec3 emissiveColor;
	alignas(16) glm::vec4 lightPosition;
};

struct StorageBufferGaussianNoise {	
	alignas(16) glm::vec4 gaussianNoise;
};

struct StorageBufferFFTAux {
	alignas(16) int index;
};

struct pushConstantsComputeH0kPipeline {
	int fourierGridSize;
	int spatialDimension;
	glm::vec2 windDirection;
	float windSpeed;
	float scalePhillips;
};

struct pushConstantsComputeHktPipeline {
	int fourierGridSize;
	int spatialDimension;
};

struct pushConstantsComputeFFTAuxPipeline {
	int fourierGridSize;
};

struct pushConstantsComputeFFTPipeline {
	int stage;
	int swap;
};

struct pushConstantsComputeHeightMapPipeline {
	int fourierGridSize;
	int swap;
};

struct UniformBufferObjectTimeHkt {	// Union buffer with the information of the object
	alignas(4) float time;
};

Camera camera;
bool keys[1024];
GLfloat lastX = 400, lastY = 300;
bool firstMouse = true;

GLfloat CameraSpeed = 0.1f;
GLfloat lastFrame = 0.0f;

bool isCursorOut = false;

// Moves/alters the camera positions based on user input
void DoMovement()
{
	if (keys[GLFW_KEY_W])
	{
		camera.ProcessKeyboard(FORWARD, CameraSpeed);
	}

	if (keys[GLFW_KEY_S])
	{
		camera.ProcessKeyboard(BACKWARD, CameraSpeed);
	}

	if (keys[GLFW_KEY_A])
	{
		camera.ProcessKeyboard(LEFT, CameraSpeed);
	}

	if (keys[GLFW_KEY_D])
	{
		camera.ProcessKeyboard(RIGHT, CameraSpeed);
	}

	if (keys[GLFW_KEY_E])
	{
		camera.ProcessKeyboard(UP, CameraSpeed);
	}

	if (keys[GLFW_KEY_Q])
	{
		camera.ProcessKeyboard(DOWN, CameraSpeed);
	}
}

// Is called whenever a key is pressed/released via GLFW
void KeyCallback(GLFWwindow *window, int key, int scancode, int action, int mode)
{
	if (GLFW_KEY_ESCAPE == key && GLFW_PRESS == action)
	{
		glfwSetWindowShouldClose(window, GL_TRUE);
	}

	if (GLFW_KEY_LEFT_ALT == key && GLFW_PRESS == action)
	{
		if (!isCursorOut) {
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		}
		else {
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
		}

		isCursorOut = !isCursorOut;
	}


	if (key >= 0 && key < 1024)
	{
		if (action == GLFW_PRESS)
		{
			keys[key] = true;
		}
		else if (action == GLFW_RELEASE)
		{
			keys[key] = false;
		}
	}
}

void MouseCallback(GLFWwindow *window, double xPos, double yPos)
{
	if (firstMouse)
	{
		lastX = float(xPos);
		lastY = float(yPos);
		firstMouse = false;
	}
	GLfloat xOffset = float(xPos) - lastX;
	GLfloat yOffset = lastY - float(yPos);  // Reversed since y-coordinates go from bottom to left

	lastX = float(xPos);
	lastY = float(yPos);
	if (!isCursorOut)
		camera.ProcessMouseMovement(xOffset, yOffset);
}


// It will store the Vulkan objects that will be initialize, used and deallocated
class HelloTriangleApplication {
public:
	void run() {	// Abstaction of all the things that need to be done before doing anything in just one call to this method.
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}

private:
	GLFWwindow* window;	// holds a pointer to the window

	VkInstance instance;	// holds the instance
	VkDebugUtilsMessengerEXT debugMessenger;	// handle debug messages
	VkSurfaceKHR surface;	// represents a surface to render images on it

	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;	// Holds the graphics card selected and it will be destroyed when the VkInstance is destroyed.
	VkDevice device;	// holds the logical device handler

	VkQueue graphicsQueue;	// handler graphics queue
	VkQueue presentQueue;	// handler present queue

	VkSwapchainKHR swapChain;	// handler of the swap chain
	std::vector<VkImage> swapChainImages;	// Array containing all the images of the swap chain
	VkFormat swapChainImageFormat;	// Handler of the format of the swap chain
	VkExtent2D swapChainExtent;	// Handler of the swap chain extent
	std::vector<VkImageView> swapChainImageViews;	// Array to handle all the views into the images of the swap chain
	std::vector<VkFramebuffer> swapChainFramebuffers;	// Array of all the frame buffers of the swap chain
	
	std::vector<VkFramebuffer> swapChainFramebuffersGUI;	// Array of all the frame buffers of the swap chain

	VkRenderPass renderPass;	// handler of the render pass
	VkRenderPass renderPassGUI; // handler of the render pass to create the GUI
	VkDescriptorSetLayout descriptorSetLayout;
	VkPipelineLayout pipelineLayout;	// handler of the pipeline layout
	VkPipeline graphicsPipeline;	// handler of the whole graphics pipelin
	
	VkPipelineLayout computeH0kPipelineLayout;	// handler of the pipeline layout
	VkPipeline computeH0kPipeline;	// handler of the whole graphics pipeline

	VkPipelineLayout computeHktPipelineLayout;	// handler of the pipeline layout
	VkPipeline computeHktPipeline;	// handler of the whole graphics pipeline

	VkPipelineLayout computeFFTAuxPipelineLayout;	// handler of the pipeline layout
	VkPipeline computeFFTAuxPipeline;	// handler of the whole graphics pipeline

	VkPipelineLayout computeFFTHorizontalPipelineLayout;	// handler of the pipeline layout
	VkPipeline computeFFTHorizontalPipeline;	// handler of the whole graphics pipeline

	VkPipelineLayout computeFFTVerticalPipelineLayout;	// handler of the pipeline layout
	VkPipeline computeFFTVerticalPipeline;	// handler of the whole graphics pipeline

	VkPipelineLayout computeHeightMapPipelineLayout;	// handler of the pipeline layout
	VkPipeline computeHeightMapPipeline;	// handler of the whole graphics pipeline

	VkPipelineLayout pipelineLayoutQuad;	// handler of the pipeline layout
	VkPipeline graphicsPipelineQuad;	// handler of the whole graphics pipeline

	VkPipelineLayout pipelineLayoutCube;	// handler of the pipeline layout
	VkPipeline graphicsPipelineCube;	// handler of the whole graphics pipeline

	VkCommandPool commandPool;	// Manages the memory that will store the buffers and command buffers
	VkCommandPool commandPoolGUI;	// Manages the memory that will store the buffers and command buffers

	VkImage maaImage;
	VkDeviceMemory maaImageMemory;
	VkImageView maaImageView;

	VkImage depthImage;	
	VkDeviceMemory depthImageMemory;
	VkImageView depthImageView;

	VkImage computeShaderImageHk0;
	VkDeviceMemory computeShaderImageMemoryHk0;
	VkImageView computeShaderImageViewHk0;

	VkImage computeShaderImageHk0minus;
	VkDeviceMemory computeShaderImageMemoryHk0minus;
	VkImageView computeShaderImageViewHk0minus;

	VkImage computeShaderImageHkt;
	VkDeviceMemory computeShaderImageMemoryHkt;
	VkImageView computeShaderImageViewHkt;

	VkImage computeShaderImageFFTAux;
	VkDeviceMemory computeShaderImageMemoryFFTAux;
	VkImageView computeShaderImageViewFFTAux;

	VkImage computeShaderImageFFTAlternate;
	VkDeviceMemory computeShaderImageMemoryFFTAlternate;
	VkImageView computeShaderImageViewFFTAlternate;

	VkImage computeShaderImageHeightMap;
	VkDeviceMemory computeShaderImageMemoryHeightMap;
	VkImageView computeShaderImageViewHeightMap;

	VkImage computeShaderImageSlopeX;
	VkDeviceMemory computeShaderImageMemorySlopeX;
	VkImageView computeShaderImageViewSlopeX;

	VkImage computeShaderImageSlopeXAlternate;
	VkDeviceMemory computeShaderImageMemorySlopeXAlternate;
	VkImageView computeShaderImageViewSlopeXAlternate;

	VkImage computeShaderImageSlopeXFinal;
	VkDeviceMemory computeShaderImageMemorySlopeXFinal;
	VkImageView computeShaderImageViewSlopeXFinal;

	VkImage computeShaderImageSlopeZ;
	VkDeviceMemory computeShaderImageMemorySlopeZ;
	VkImageView computeShaderImageViewSlopeZ;

	VkImage computeShaderImageSlopeZAlternate;
	VkDeviceMemory computeShaderImageMemorySlopeZAlternate;
	VkImageView computeShaderImageViewSlopeZAlternate;

	VkImage computeShaderImageSlopeZFinal;
	VkDeviceMemory computeShaderImageMemorySlopeZFinal;
	VkImageView computeShaderImageViewSlopeZFinal;

	VkImage computeShaderImageDispX;
	VkDeviceMemory computeShaderImageMemoryDispX;
	VkImageView computeShaderImageViewDispX;

	VkImage computeShaderImageDispXAlternate;
	VkDeviceMemory computeShaderImageMemoryDispXAlternate;
	VkImageView computeShaderImageViewDispXAlternate;

	VkImage computeShaderImageDispXFinal;
	VkDeviceMemory computeShaderImageMemoryDispXFinal;
	VkImageView computeShaderImageViewDispXFinal;

	VkImage computeShaderImageDispZ;
	VkDeviceMemory computeShaderImageMemoryDispZ;
	VkImageView computeShaderImageViewDispZ;

	VkImage computeShaderImageDispZAlternate;
	VkDeviceMemory computeShaderImageMemoryDispZAlternate;
	VkImageView computeShaderImageViewDispZAlternate;

	VkImage computeShaderImageDispZFinal;
	VkDeviceMemory computeShaderImageMemoryDispZFinal;
	VkImageView computeShaderImageViewDispZFinal;

	VkImage textureImage;
	VkDeviceMemory textureImageMemory;
	VkImageView textureImageView;
	VkSampler textureSamplerCube;
	VkSampler textureSampler;

	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
	VkBuffer vertexBuffer;
	VkDeviceMemory vertexBufferMemory;
	VkBuffer indexBuffer;
	VkDeviceMemory indexBufferMemory;


	std::vector<Vertex> verticesQuad;
	std::vector<uint32_t> indicesQuad;
	VkBuffer vertexBufferQuad;
	VkDeviceMemory vertexBufferMemoryQuad;
	VkBuffer indexBufferQuad;
	VkDeviceMemory indexBufferMemoryQuad;

	std::vector<VertexOnlyPos> verticesCube;
	std::vector<uint32_t> indicesCube;
	VkBuffer vertexBufferCube;
	VkDeviceMemory vertexBufferMemoryCube;
	VkBuffer indexBufferCube;
	VkDeviceMemory indexBufferMemoryCube;


	std::vector<VkBuffer> uniformBuffers;
	std::vector<VkDeviceMemory> uniformBuffersMemory;

	std::vector<VkBuffer> lightUniformBuffers;
	std::vector<VkDeviceMemory> lightUniformBuffersMemory;

	VkBuffer gaussianNoiseStorageBuffers;
	VkDeviceMemory gaussianNoiseStorageBuffersMemory; 

	std::vector<VkBuffer> uniformBuffersTimeHkt;
	std::vector<VkDeviceMemory> uniformBuffersTimeHktMemory;

	VkBuffer FFTAuxStorageBuffers;
	VkDeviceMemory FFTAuxStorageBuffersMemory;

	VkDescriptorPool descriptorPool;
	VkDescriptorPool descriptorPoolGUI;
	std::vector<VkDescriptorSet> descriptorSets;

	std::vector<VkCommandBuffer> commandBuffers;	// Array of a command buffer per image in the swap chain
	std::vector<VkCommandBuffer> commandBuffersGUI;	// Array of a command buffer per image in the swap chain
	std::vector<VkSemaphore> imageAvailableSemaphores;	// Array of semaphores to indicate that an image has been acquire and is ready for rendering
	std::vector<VkSemaphore> renderFinishedSemaphores;	// Array of semaphores to indicate that rendering has finished and presentation can happend
	std::vector<VkFence> inFlightFences;	// Array of all the fences that will be simultaneously built
	std::vector<VkFence> imagesInFlight;	// Array of the fences that each image will have
	size_t currentFrame = 0;
	
	
	bool framebufferResized = false;	// Tells if the framebuffer has been resized

	std::default_random_engine generator;
	std::normal_distribution<double> distribution{ 0.0, 1.0 };
	
	std::array<int, 1> pushConstantsRenderScenePipeline;	// variables to control the behaviour of the GUI
	std::array<int, 11> pushConstantsQuadPipeline;

	bool apply = false;
	bool isDisplayTextures = false;
	
	bool isH0k = false;
	bool isH0kMinus = false;
	bool isHkt = false;
	bool isFFTAux = false;
	bool isHorizontalFFT = false;
	bool isVerticalFFT = false;
	bool isHeightMap = false;
	bool isWavy = false;
	bool isSlopeX = false;
	bool isSlopeZ = false;
	bool isDispX = false;
	bool isDispZ = false;
	bool isTimeStop = false;
	
	int fourierGridSize = 256;
	int spatialDimension = 1300;
	glm::vec2 windDirection = glm::vec2(1.0, 1.0);
	float windSpeed = 40;
	float scalePhillips = 4;
	float timeAnimation = 0;
	float waterSpeed = 1.6f;
	glm::vec4 lightPosition = glm::vec4(500.0f, 500.0f, 500.0f, 1.0f);
	glm::vec3 waterColor = glm::vec3(0.047f, 0.0594f, 0.301f);
	
	VkSampleCountFlagBits numberSamplesRemoveAliasing = VK_SAMPLE_COUNT_1_BIT;
	bool aliasing = true;

	// Start GLFW and creates the window
	void initWindow() {
		glfwInit();	// Starts GLFW

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);	// It tells GLFW that it needs a window but is not an OpenGL one.

		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr); // Create the actual window with the specified Width and Height, named Vulkan
																			  // and the first nullptr is to decide which monitor to use and the second one is only for OpenGl
		glfwSetWindowUserPointer(window, this);
		glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);	// Allows GLFW to use the method specified in case that the window is tryed to be resized
	}

	// Uses the GLFW function to make feasible to resize the window and set that it has been resized
	static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
		auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
		app->framebufferResized = true;
	}

	void initVulkan() {	// intialize all Vulkan Objects
		createInstance();	// Create the instance which is the way to communicate between our application and the Vulkan library
		setupDebugMessenger();	// Set up the debugger to properly show messages
		createSurface();		// Creates a surface to render images into it depending of the device where it is running
		pickPhysicalDevice();	// Selects a number of graphics cards to use
		createLogicalDevice();	// Creates queues from the ones available to the system
		createSwapChain();	// Creates the swap chain implementation
		createImageViews();	// Creates the image views for the images of the swap chain
		createRenderPass();	// Specifies how many color and depth buffers and how they should be configured 
		createRenderPassGUI();
		createDescriptorSetLayout();
		createGraphicsPipeline();	// Creates the whole Graphics pipeline with all the shaders
		createcomputeH0kPipeline();	// Creates the whole Graphics pipeline with all the shaders
		createcomputeHktPipeline();
		createcomputeFFTAuxPipeline();
		createcomputeFFTHorizontalPipeline();
		createcomputeFFTVerticalPipeline();
		createcomputeHeightMapPipeline();
		createGraphicsPipelineQuad();
		createGraphicsPipelineCube();
		createCommandPool();	// Create a manager of the memory that will store the buffers and command buffers
		createMaaResources();
		createDepthResources();
		createComputeShaderResources();
		createFramebuffers();	// Actually create the FrameBuffers with all the attachments and it will hold all the images of the swap chain
		createFramebuffersGUI();
		createTextureImage();	// Creates the texture image by reading a file and allocates space
		createTextureImageView(); // Creates an image view for the texture
		createTextureSampler();	// Create a sampler for the texture
		createTextureSamplerCube();	// Create a sampler for the texture
		loadModel();	// Loads the obj file containing the model
		createVertexBuffer();	// Creates the vertex buffers that will hold the vertices data
		createVertexBufferQuad();
		createVertexBufferCube();
		createIndexBuffer();	// Creates an indices buffer to hold the indices of the vertices that will be used
		createIndexBufferQuad();
		createIndexBufferCube();
		createUniformBuffers();	// Creates the object uniform buffer
		createLightUniformBuffers();	// Creates the light uniform buffer
		creategaussianNoiseStorageBuffers();
		createIndicesFFTAuxStorageBuffers();
		createUniformBuffersTimeHkt();
		createDescriptorPool();	// Create descriptor pools for the uniform buffer
		createDescriptorPoolGUI();
		createDescriptorSets();	// Create descriptor sets for the uniform buffer
		createPreprocessCommandBuffers();	// Creates a command buffer per image in the swap chain to be able to bind the correct buffer during the draw call
		createCommandBuffers();	// Creates a command buffer per image in the swap chain to be able to bind the correct buffer during the draw call
		createSyncObjects();	// Creates all the objects required to create synchronization
	}

	void mainLoop() {	// loops untill the window is closed
		
		ImGui::CreateContext();	// Set the environment for the gui

		ImGui::StyleColorsDark();
		ImGuiIO& io = ImGui::GetIO(); (void)io;

		ImGui_ImplGlfw_InitForVulkan(window, true);
		ImGui_ImplVulkan_InitInfo init_info = {};	// expose values of vulkan to the gui
		init_info.Instance = instance;
		init_info.PhysicalDevice = physicalDevice;
		init_info.Device = device;
		init_info.Allocator = nullptr;
		init_info.QueueFamily = 4;
		init_info.Queue = graphicsQueue;
		init_info.PipelineCache = VK_NULL_HANDLE;
		init_info.DescriptorPool = descriptorPoolGUI;
		init_info.MinImageCount = MAX_FRAMES_IN_FLIGHT;
		init_info.ImageCount = uint32_t(swapChainImages.size());
		init_info.CheckVkResultFn = nullptr;
		ImGui_ImplVulkan_Init(&init_info, renderPassGUI);

		VkCommandBuffer command_buffer = beginSingleTimeCommands();
		ImGui_ImplVulkan_CreateFontsTexture(command_buffer);
		endSingleTimeCommands(command_buffer);


		glfwSetKeyCallback(window, KeyCallback);
		glfwSetCursorPosCallback(window, MouseCallback);

		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

		bool prevDisplayTextures = false;	// variables to control the behaviour of the gui (set to default)
		bool prevH0k = false;
		bool prevH0kMinus = false;
		bool prevHkt = false;
		bool prevFFTAux = false;
		bool prevHorizontalFFT = false;
		bool prevVerticalFFT = false;
		bool prevHeightMap = false;
		bool prevWavy = false;
		bool prevSlopeX = false;
		bool prevSlopeZ = false;
		bool prevDispX = false;
		bool prevDispZ = false;
		bool prevIsTimeStop = false;
		int prevFourierGridSize = fourierGridSize;
		int prevSpatialDimension = spatialDimension;
		glm::vec2 prevWindDirection = windDirection;
		float prevWindSpeed = windSpeed;
		float prevScalePhillips = scalePhillips;
		glm::vec3 prevWaterColor = waterColor;
		bool prevAliasing = aliasing;


		while (!glfwWindowShouldClose(window)) {	// Checks if the windows has closed and loops untill it does.
			glfwPollEvents();	// ask for events
			if (!isCursorOut)
				DoMovement();
			ImGui_ImplVulkan_NewFrame();	// create a new frame for the gui
			ImGui_ImplGlfw_NewFrame();
			ImGui::NewFrame();
			ImGui::Checkbox("apply", &apply);
			ImGui::Checkbox("display textures", &isDisplayTextures);
			ImGui::Checkbox("display h0k", &isH0k);
			ImGui::Checkbox("display h0kMinus", &isH0kMinus);
			ImGui::Checkbox("display h(k,t)", &isHkt);
			ImGui::Checkbox("display IFFT aux", &isFFTAux);
			ImGui::Checkbox("display Horizontal IFFT", &isHorizontalFFT);
			ImGui::Checkbox("display Vertical IFFT", &isVerticalFFT);
			ImGui::Checkbox("display HeightMap", &isHeightMap);
			ImGui::Checkbox("display SlopeX", &isSlopeX);
			ImGui::Checkbox("display SlopeZ", &isSlopeZ);
			ImGui::Checkbox("display DipsX", &isDispX);
			ImGui::Checkbox("display DispZ", &isDispZ);
			ImGui::Checkbox("Change Choppy", &isWavy);
			ImGui::Checkbox("Stop time", &isTimeStop);
			ImGui::Checkbox("remove Aliasing", &aliasing);
			ImGui::InputFloat3("Sun position", &lightPosition.x, 2);
			ImGui::InputFloat3("Water color", &waterColor.x, 4);
			ImGui::InputInt("fourierGridSize", &fourierGridSize);
			ImGui::InputInt("spatialDimension", &spatialDimension);
			ImGui::InputFloat3("wind direction", &windDirection.x,2);
			ImGui::InputFloat("wind speed", &windSpeed);
			ImGui::InputFloat("scale Phillips", &scalePhillips);
			ImGui::InputFloat("water speed", &waterSpeed);
			ImGui::Text("Application average %.3f ms/frame (%.5f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
			ImGui::Render();

			// If there is a change in the gui
			if ((isDisplayTextures != prevDisplayTextures || isH0k != prevH0k || isH0kMinus != prevH0kMinus
				|| isHkt != prevHkt || isFFTAux != prevFFTAux || isHorizontalFFT != prevHorizontalFFT || isVerticalFFT != prevVerticalFFT
				|| isHeightMap != prevHeightMap || isWavy != prevWavy || isSlopeX != prevSlopeX || isSlopeZ != prevSlopeZ || isDispX != prevDispX || isDispZ != prevDispZ
				|| isTimeStop != prevIsTimeStop || fourierGridSize != prevFourierGridSize || spatialDimension != prevSpatialDimension
				|| windDirection.x != prevWindDirection.x || windDirection.y != prevWindDirection.y || windSpeed != prevWindSpeed
				|| scalePhillips != prevScalePhillips || waterColor.x != prevWaterColor.x || waterColor.y != prevWaterColor.y || waterColor.z != prevWaterColor.z) && apply)
			{
				prevDisplayTextures = isDisplayTextures;
				prevH0k = isH0k;
				prevH0kMinus = isH0kMinus;
				prevHkt = isHkt;
				prevFFTAux = isFFTAux;
				prevHorizontalFFT = isHorizontalFFT;
				prevVerticalFFT = isVerticalFFT;
				prevHeightMap = isHeightMap;
				prevWavy = isWavy;
				prevSlopeX = isSlopeX;
				prevSlopeZ = isSlopeZ;
				prevDispX = isDispX;
				prevDispZ = isDispZ;
				prevIsTimeStop = isTimeStop;
				prevFourierGridSize = fourierGridSize;
				prevSpatialDimension = spatialDimension;
				prevWindDirection = windDirection;
				prevWaterColor = waterColor;
				prevWindSpeed = windSpeed;
				prevScalePhillips = scalePhillips;
				apply = false;

				recreateSwapChain();
			}

			if (aliasing != prevAliasing)
			{
				prevAliasing = aliasing;
				VkSampleCountFlagBits aux = msaaSamples;
				msaaSamples = numberSamplesRemoveAliasing;
				numberSamplesRemoveAliasing = aux;
				recreateSwapChain();
			}

			drawFrame();	// do all the work untill a frame is drawn
		}
		vkDeviceWaitIdle(device);	// Waits untill the logical device finish all its tasks to be able to destroy the windows later on
	}

	// Handles all the clean up related with the swap chain
	void cleanupSwapChain() {
		vkDestroyImageView(device, maaImageView, nullptr);
		vkDestroyImage(device, maaImage, nullptr);
		vkFreeMemory(device, maaImageMemory, nullptr);

		vkDestroyImageView(device, depthImageView, nullptr);
		vkDestroyImage(device, depthImage, nullptr);
		vkFreeMemory(device, depthImageMemory, nullptr);

		vkDestroyImageView(device, computeShaderImageViewHk0, nullptr);
		vkDestroyImage(device, computeShaderImageHk0, nullptr);
		vkFreeMemory(device, computeShaderImageMemoryHk0, nullptr);

		vkDestroyImageView(device, computeShaderImageViewHk0minus, nullptr);
		vkDestroyImage(device, computeShaderImageHk0minus, nullptr);
		vkFreeMemory(device, computeShaderImageMemoryHk0minus, nullptr);

		vkDestroyImageView(device, computeShaderImageViewHkt, nullptr);
		vkDestroyImage(device, computeShaderImageHkt, nullptr);
		vkFreeMemory(device, computeShaderImageMemoryHkt, nullptr);

		vkDestroyImageView(device, computeShaderImageViewFFTAux, nullptr);
		vkDestroyImage(device, computeShaderImageFFTAux, nullptr);
		vkFreeMemory(device, computeShaderImageMemoryFFTAux, nullptr);

		vkDestroyImageView(device, computeShaderImageViewFFTAlternate, nullptr);
		vkDestroyImage(device, computeShaderImageFFTAlternate, nullptr);
		vkFreeMemory(device, computeShaderImageMemoryFFTAlternate, nullptr);

		vkDestroyImageView(device, computeShaderImageViewHeightMap, nullptr);
		vkDestroyImage(device, computeShaderImageHeightMap, nullptr);
		vkFreeMemory(device, computeShaderImageMemoryHeightMap, nullptr);

		vkDestroyImageView(device, computeShaderImageViewSlopeX, nullptr);
		vkDestroyImage(device, computeShaderImageSlopeX, nullptr);
		vkFreeMemory(device, computeShaderImageMemorySlopeX, nullptr);

		vkDestroyImageView(device, computeShaderImageViewSlopeXAlternate, nullptr);
		vkDestroyImage(device, computeShaderImageSlopeXAlternate, nullptr);
		vkFreeMemory(device, computeShaderImageMemorySlopeXAlternate, nullptr);

		vkDestroyImageView(device, computeShaderImageViewSlopeXFinal, nullptr);
		vkDestroyImage(device, computeShaderImageSlopeXFinal, nullptr);
		vkFreeMemory(device, computeShaderImageMemorySlopeXFinal, nullptr);

		vkDestroyImageView(device, computeShaderImageViewSlopeZ, nullptr);
		vkDestroyImage(device, computeShaderImageSlopeZ, nullptr);
		vkFreeMemory(device, computeShaderImageMemorySlopeZ, nullptr);

		vkDestroyImageView(device, computeShaderImageViewSlopeZAlternate, nullptr);
		vkDestroyImage(device, computeShaderImageSlopeZAlternate, nullptr);
		vkFreeMemory(device, computeShaderImageMemorySlopeZAlternate, nullptr);

		vkDestroyImageView(device, computeShaderImageViewSlopeZFinal, nullptr);
		vkDestroyImage(device, computeShaderImageSlopeZFinal, nullptr);
		vkFreeMemory(device, computeShaderImageMemorySlopeZFinal, nullptr);

		vkDestroyImageView(device, computeShaderImageViewDispX, nullptr);
		vkDestroyImage(device, computeShaderImageDispX, nullptr);
		vkFreeMemory(device, computeShaderImageMemoryDispX, nullptr);

		vkDestroyImageView(device, computeShaderImageViewDispXAlternate, nullptr);
		vkDestroyImage(device, computeShaderImageDispXAlternate, nullptr);
		vkFreeMemory(device, computeShaderImageMemoryDispXAlternate, nullptr);

		vkDestroyImageView(device, computeShaderImageViewDispXFinal, nullptr);
		vkDestroyImage(device, computeShaderImageDispXFinal, nullptr);
		vkFreeMemory(device, computeShaderImageMemoryDispXFinal, nullptr);

		vkDestroyImageView(device, computeShaderImageViewDispZ, nullptr);
		vkDestroyImage(device, computeShaderImageDispZ, nullptr);
		vkFreeMemory(device, computeShaderImageMemoryDispZ, nullptr);

		vkDestroyImageView(device, computeShaderImageViewDispZAlternate, nullptr);
		vkDestroyImage(device, computeShaderImageDispZAlternate, nullptr);
		vkFreeMemory(device, computeShaderImageMemoryDispZAlternate, nullptr);

		vkDestroyImageView(device, computeShaderImageViewDispZFinal, nullptr);
		vkDestroyImage(device, computeShaderImageDispZFinal, nullptr);
		vkFreeMemory(device, computeShaderImageMemoryDispZFinal, nullptr);

		for (auto framebuffer : swapChainFramebuffers) {	// for every framebuffer aka. images in the swap chain
			vkDestroyFramebuffer(device, framebuffer, nullptr);	// destroys the frame buffer
		}

		for (auto framebuffer : swapChainFramebuffersGUI) {	// for every framebuffer aka. images in the swap chain
			vkDestroyFramebuffer(device, framebuffer, nullptr);	// destroys the frame buffer
		}

		vkFreeCommandBuffers(device, commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());	// Clean up the existing command buffers without destroying them
		vkFreeCommandBuffers(device, commandPoolGUI, static_cast<uint32_t>(commandBuffersGUI.size()), commandBuffersGUI.data());	// Clean up the existing command buffers without destroying them
		
		vkDestroyPipeline(device, graphicsPipeline, nullptr);	// destroys the handler for the graphics pipeline
		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);	// destroys the handler for the pipeline layout

		vkDestroyPipeline(device, computeH0kPipeline, nullptr);	// destroys the handler for the graphics pipeline
		vkDestroyPipelineLayout(device, computeH0kPipelineLayout, nullptr);	// destroys the handler for the pipeline layout

		vkDestroyPipeline(device, computeHktPipeline, nullptr);	// destroys the handler for the graphics pipeline
		vkDestroyPipelineLayout(device, computeHktPipelineLayout, nullptr);	// destroys the handler for the pipeline layout

		vkDestroyPipeline(device, computeFFTAuxPipeline, nullptr);	// destroys the handler for the graphics pipeline
		vkDestroyPipelineLayout(device, computeFFTAuxPipelineLayout, nullptr);	// destroys the handler for the pipeline layout

		vkDestroyPipeline(device, computeFFTHorizontalPipeline, nullptr);	// destroys the handler for the graphics pipeline
		vkDestroyPipelineLayout(device, computeFFTHorizontalPipelineLayout, nullptr);	// destroys the handler for the pipeline layout

		vkDestroyPipeline(device, computeFFTVerticalPipeline, nullptr);	// destroys the handler for the graphics pipeline
		vkDestroyPipelineLayout(device, computeFFTVerticalPipelineLayout, nullptr);	// destroys the handler for the pipeline layout

		vkDestroyPipeline(device, computeHeightMapPipeline, nullptr);	// destroys the handler for the graphics pipeline
		vkDestroyPipelineLayout(device, computeHeightMapPipelineLayout, nullptr);	// destroys the handler for the pipeline layout

		vkDestroyPipeline(device, graphicsPipelineQuad, nullptr);	// destroys the handler for the graphics pipeline
		vkDestroyPipelineLayout(device, pipelineLayoutQuad, nullptr);	// destroys the handler for the pipeline layout

		vkDestroyPipeline(device, graphicsPipelineCube, nullptr);	// destroys the handler for the graphics pipeline
		vkDestroyPipelineLayout(device, pipelineLayoutCube, nullptr);	// destroys the handler for the pipeline layout
		
		vkDestroyRenderPass(device, renderPass, nullptr);	// destroys the handler for the render pass
		vkDestroyRenderPass(device, renderPassGUI, nullptr);
		for (auto imageView : swapChainImageViews) {	// for every image view
			vkDestroyImageView(device, imageView, nullptr);	// destroy it 
		}

		vkDestroySwapchainKHR(device, swapChain, nullptr);	// detroys the swap chain 

		for (size_t i = 0; i < swapChainImages.size(); i++) {
			vkDestroyBuffer(device, uniformBuffers[i], nullptr);
			vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
			vkDestroyBuffer(device, lightUniformBuffers[i], nullptr);
			vkFreeMemory(device, lightUniformBuffersMemory[i], nullptr);
			vkDestroyBuffer(device, uniformBuffersTimeHkt[i], nullptr);
			vkFreeMemory(device, uniformBuffersTimeHktMemory[i], nullptr);
		}

		vkDestroyBuffer(device, gaussianNoiseStorageBuffers, nullptr);
		vkFreeMemory(device, gaussianNoiseStorageBuffersMemory, nullptr);

		vkDestroyBuffer(device, FFTAuxStorageBuffers, nullptr);
		vkFreeMemory(device, FFTAuxStorageBuffersMemory, nullptr);

		ImGui_ImplVulkan_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();

		vkDestroyDescriptorPool(device, descriptorPoolGUI, nullptr);
		vkDestroyDescriptorPool(device, descriptorPool, nullptr);

	}

	void cleanup() {	 // deallocate every Vulkan Object

		cleanupSwapChain();

		vkDestroySampler(device, textureSampler, nullptr);
		vkDestroySampler(device, textureSamplerCube, nullptr);
		vkDestroyImageView(device, textureImageView, nullptr);

		vkDestroyImage(device, textureImage, nullptr);
		vkFreeMemory(device, textureImageMemory, nullptr);

		vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

		vkDestroyBuffer(device, indexBuffer, nullptr);
		vkFreeMemory(device, indexBufferMemory, nullptr);
		vkDestroyBuffer(device, vertexBuffer, nullptr);
		vkFreeMemory(device, vertexBufferMemory, nullptr);

		vkDestroyBuffer(device, indexBufferQuad, nullptr);
		vkFreeMemory(device, indexBufferMemoryQuad, nullptr);
		vkDestroyBuffer(device, vertexBufferQuad, nullptr);
		vkFreeMemory(device, vertexBufferMemoryQuad, nullptr);
	
		vkDestroyBuffer(device, indexBufferCube, nullptr);
		vkFreeMemory(device, indexBufferMemoryCube, nullptr);
		vkDestroyBuffer(device, vertexBufferCube, nullptr);
		vkFreeMemory(device, vertexBufferMemoryCube, nullptr);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);	// Destroy the semaphores per the amount of simultaneous frames allowed
			vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
			vkDestroyFence(device, inFlightFences[i], nullptr);	// Destroy the fence per the amount of simultaneous frames allowed
		}

		vkDestroyCommandPool(device, commandPool, nullptr);	// Destroys the command pool buffer
		vkDestroyCommandPool(device, commandPoolGUI, nullptr);	// Destroys the command pool buffer

		vkDestroyDevice(device, nullptr);	// Destroys the logical device handler and so it cleans the related objects to it as the handlers for the queues 

		if (enableValidationLayers) {	// if validations are activated, destroy the debugMessage object
			DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
		}

		vkDestroySurfaceKHR(instance, surface, nullptr); // GLFW doesn't give us a function to destroy the surface, so we use the ones of the original API
		vkDestroyInstance(instance, nullptr);	// destroy the instance

		glfwDestroyWindow(window);	// destroy the window

		glfwTerminate();	// tells GLFW to finish
	}

	// Restart the swap chain in the case that there is any update related with the window
	void recreateSwapChain() {
		int width = 0, height = 0;
		glfwGetFramebufferSize(window, &width, &height);	// Gather the sizes of the window with the help of GLFW
		while (width == 0 || height == 0) {	// While the sizes of the windows are minimize
			glfwGetFramebufferSize(window, &width, &height);
			glfwWaitEvents();	// Wait for events that change them
		}

		vkDeviceWaitIdle(device);	// We wait until the logical device finishes all its tasks

		cleanupSwapChain();	// Clean up all the previous information of the swap chain before reconstructing it again

		// do all the configurations again
		createSwapChain();
		createImageViews();
		createRenderPass();
		createRenderPassGUI();
		createGraphicsPipeline();
		createcomputeH0kPipeline();
		createcomputeHktPipeline();
		createcomputeFFTAuxPipeline();
		createcomputeFFTHorizontalPipeline();
		createcomputeFFTVerticalPipeline();
		createcomputeHeightMapPipeline();
		createGraphicsPipelineQuad();
		createGraphicsPipelineCube();
		createMaaResources();
		createDepthResources();
		createComputeShaderResources();
		createFramebuffers();
		createFramebuffersGUI();
		loadModel();
		createVertexBuffer();	// Creates the vertex buffers that will hold the vertices data
		createVertexBufferQuad();
		createVertexBufferCube();
		createIndexBuffer();	// Creates an indices buffer to hold the indices of the vertices that will be used
		createIndexBufferQuad();
		createIndexBufferCube();
		createUniformBuffers();
		createLightUniformBuffers();
		creategaussianNoiseStorageBuffers();
		createIndicesFFTAuxStorageBuffers();
		createUniformBuffersTimeHkt();
		createDescriptorPool();
		createDescriptorPoolGUI();
		createDescriptorSets();
		createPreprocessCommandBuffers();
		createCommandBuffers();

		ImGui::CreateContext();	// recreate the gui

		ImGui::StyleColorsDark();
		ImGuiIO& io = ImGui::GetIO(); (void)io;

		ImGui_ImplGlfw_InitForVulkan(window, true);
		ImGui_ImplVulkan_InitInfo init_info = {};
		init_info.Instance = instance;
		init_info.PhysicalDevice = physicalDevice;
		init_info.Device = device;
		init_info.Allocator = nullptr;
		init_info.QueueFamily = 4;
		init_info.Queue = graphicsQueue;
		init_info.PipelineCache = VK_NULL_HANDLE;
		init_info.DescriptorPool = descriptorPoolGUI;
		init_info.MinImageCount = MAX_FRAMES_IN_FLIGHT;
		init_info.ImageCount = uint32_t(swapChainImages.size());
		init_info.CheckVkResultFn = nullptr;
		ImGui_ImplVulkan_Init(&init_info, renderPassGUI);

		glfwSetKeyCallback(window, KeyCallback);
		glfwSetCursorPosCallback(window, MouseCallback);

		VkCommandBuffer command_buffer = beginSingleTimeCommands();
		ImGui_ImplVulkan_CreateFontsTexture(command_buffer);
		endSingleTimeCommands(command_buffer);

		ImGui_ImplVulkan_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
		
		ImGui::Checkbox("apply", &apply);
		ImGui::Checkbox("display textures", &isDisplayTextures);
		ImGui::Checkbox("display h0k", &isH0k);
		ImGui::Checkbox("display h0kMinus", &isH0kMinus);
		ImGui::Checkbox("display h(k,t)", &isHkt);
		ImGui::Checkbox("display IFFT aux", &isFFTAux);
		ImGui::Checkbox("display Horizontal IFFT", &isHorizontalFFT);
		ImGui::Checkbox("display Vertical IFFT", &isVerticalFFT);
		ImGui::Checkbox("display HeightMap", &isHeightMap);
		ImGui::Checkbox("display SlopeX", &isSlopeX);
		ImGui::Checkbox("display SlopeZ", &isSlopeZ);
		ImGui::Checkbox("display DipsX", &isDispX);
		ImGui::Checkbox("display DispZ", &isDispZ);
		ImGui::Checkbox("Change Choppy", &isWavy);
		ImGui::Checkbox("Stop time", &isTimeStop);
		ImGui::Checkbox("remove Aliasing", &aliasing);
		ImGui::InputFloat3("Sun position", &lightPosition.x, 2);
		ImGui::InputFloat3("Water color", &waterColor.x, 4);
		ImGui::InputInt("fourierGridSize", &fourierGridSize);
		ImGui::InputInt("spatialDimension", &spatialDimension);
		ImGui::InputFloat3("wind direction", &windDirection.x, 2);
		ImGui::InputFloat("wind speed", &windSpeed);
		ImGui::InputFloat("scale Phillips", &scalePhillips);
		ImGui::InputFloat("water speed", &waterSpeed);
		ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

		ImGui::Render();

		createCommandPoolGUI(VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
		createCommandBuffersGUI();

	}

	void createInstance() {
		if (enableValidationLayers && !checkValidationLayerSupport()) {	// check if the validation layer are activated and if all of the desired ones are in the system.
																		// if not, throw a runtime error
			throw std::runtime_error("validation layers requested, but not available!");
		}

		VkApplicationInfo appInfo = {};
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;			// type or identifier of the struct
		appInfo.pApplicationName = "Hello Triangle";				// gives the name to the application
		appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);		// number of the version of the created application
		appInfo.pEngineName = "No Engine";							// name of the engine that creates the application
		appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);			// number of the version of the engine 
		appInfo.apiVersion = VK_API_VERSION_1_0;					// is the highest version of Vulkan the application is going to be allowed to run on.

		VkInstanceCreateInfo createInfo = {};	// it contains mandatory information to create the instance
		createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;	// type or identifier of the struct
		createInfo.pApplicationInfo = &appInfo;	// helps implementations recognize behavior inherent to applications

		auto extensions = getRequiredExtensions(); // call to the private method that controls which extensions are going to be used
		createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());	// number of global extensions enabled
		createInfo.ppEnabledExtensionNames = extensions.data();	// contains the names of the extensions enabled

		VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo;
		if (enableValidationLayers) {	// if the validation layers check is activated
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());	// number of global layers enabled
			createInfo.ppEnabledLayerNames = validationLayers.data();	// contains the names of the layers enabled

			populateDebugMessengerCreateInfo(debugCreateInfo); // create the debugger with the information in createInfo	
			createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;	// pointer to define the extension of the original structure
		}
		else {
			createInfo.enabledLayerCount = 0;	// set the number of global layers to 0

			createInfo.pNext = nullptr;
		}

		if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {	// Checks if the creation of the instance has been successful or not
			throw std::runtime_error("failed to create instance!");
		}
	}

	// Abstraction to allow the creation of two different debuggers for vkCreateInstance and vkDestroyInstance
	void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
		createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;	// type or identifier of the struct
		createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;	// Specify level of severity
		createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT; // notify the contents of the callback
		createInfo.pfnUserCallback = debugCallback;	// pointer to the callback function 
	}

	void setupDebugMessenger() {
		if (!enableValidationLayers) return;	// if the validation layer is not activated then don't do anything

		VkDebugUtilsMessengerCreateInfoEXT createInfo;	// create a struct to create the messenger
		populateDebugMessengerCreateInfo(createInfo);	// create the debugger with the information in createInfo

		// function to create the VkDebugUtilsMessengerEXT object which returns if it was a success or not
		if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
			throw std::runtime_error("failed to set up debug messenger!");	// throw and error
		}
	}

	// Creates a surface to render images into it depending of the device where it is running
	void createSurface() {
		if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {	// uses the GLFW library to create a window surface for us
			throw std::runtime_error("failed to create window surface!");
		}
	}

	// Selects which graphics cards are going to be used
	void pickPhysicalDevice() {
		uint32_t deviceCount = 0;
		vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);	// query for the number of graphics card available

		if (deviceCount == 0) {	// if there is not any graphics card that supports vulkan throw a runtime error.
			throw std::runtime_error("failed to find GPUs with Vulkan support!");
		}

		std::vector<VkPhysicalDevice> devices(deviceCount);	// array that will hold the details of every graphics card available to the system
		vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data()); // fill in the array with the contents of each graphic cards that supports Vulkan

		for (const auto& device : devices) {	// check for every available graphics card
			if (isDeviceSuitable(device)) {		// if it is suitable for our requirements if so select it and stop serching
				physicalDevice = device;
				msaaSamples = getMaxUsableSampleCount();
				break;
			}
		}

		if (physicalDevice == VK_NULL_HANDLE) {	// If any of them was suitable throw a runtime error
			throw std::runtime_error("failed to find a suitable GPU!");
		}
	}

	// Creates the logical device selecting which of the available queues is going to be used
	void createLogicalDevice() {
		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);	// retrieve the indices of the available queues

		std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
		std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() };	// get the values of the different queues available

		float queuePriority = 1.0f;
		for (uint32_t queueFamily : uniqueQueueFamilies) {	// loop through the queues available and set details for each of them
			VkDeviceQueueCreateInfo queueCreateInfo = {};
			queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;	// type or identifier of the struct
			queueCreateInfo.queueFamilyIndex = queueFamily;	// set the value of this particular queue to be created on the device
			queueCreateInfo.queueCount = 1;	// specify the number of queues to create
			queueCreateInfo.pQueuePriorities = &queuePriority;	// pointer to specify properties that apply to each created queue
			queueCreateInfos.push_back(queueCreateInfo);	// set the values configured by each queue
		}

		VkPhysicalDeviceFeatures deviceFeatures = {};
		deviceFeatures.samplerAnisotropy = VK_TRUE;
		deviceFeatures.shaderFloat64 = VK_TRUE;

		VkDeviceCreateInfo createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;	// type or identifier of the struct

		createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());	// specify the size of the queue
		createInfo.pQueueCreateInfos = queueCreateInfos.data();	// set the data of the queue

		createInfo.pEnabledFeatures = &deviceFeatures;	// pointer to enable the configuration made from the struct

		createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()); //	number of device extensions to enable
		createInfo.ppEnabledExtensionNames = deviceExtensions.data();	// data of the extensions enabled

		if (enableValidationLayers) {	// if the validation layers are activated. This is not really need it for lastest versions of Vulkan, 
										// because now there are not distinctions between instance and device validation layers
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());	// set the number of validation layers enabled
			createInfo.ppEnabledLayerNames = validationLayers.data();	// set the data of those layers that are enabled
		}
		else {
			createInfo.enabledLayerCount = 0;	// otherwise there are not any layers
		}

		if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {	// instantiate the logical device with the data of the structs and throw an error if it fails
			throw std::runtime_error("failed to create logical device!");
		}

		// the input of this functions are logical device, queue family, queue index and the pointer to return the handler 
		vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);	// retrieve queue handle for the graphics queue
		vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);		// retrieve queue handle for the present queue
	}

	// Creates the whole swap chain which are the way in which the images are shown in the screen and how it synchronize with the refresh rate of the screen
	void createSwapChain() {
		SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);	// retrieves the neccessary information to create the swap chain

		VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);	// select a surface format
		VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);	// select a present mode
		VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);	// select a swap extent

		uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;	// specify the number of images it needs to function, to avoid driver waiting we use 1 more
																				// than the minimum time
		if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {	// Check that we don't exceed the maximum number of images
																															// Where 0 in this case means there is not a maximum number of images
			imageCount = swapChainSupport.capabilities.maxImageCount;
		}

		VkSwapchainCreateInfoKHR createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;	// type of the structure
		createInfo.surface = surface;	// surface in which the swap chain will present images

		createInfo.minImageCount = imageCount;	// min number of images that the application needs
		createInfo.imageFormat = surfaceFormat.format;	// set the format of the swap chain
		createInfo.imageColorSpace = surfaceFormat.colorSpace;	// set the color space
		createInfo.imageExtent = extent;	// set the image extent
		createInfo.imageArrayLayers = 1;	// is the number of views in a multiview/stereo surface
		createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;	// describe the intended usage of the images in the swap chain

		QueueFamilyIndices indices = findQueueFamilies(physicalDevice);	// query the indices of the queues in the system
		uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };	// array of the two different queues we have

		if (indices.graphicsFamily != indices.presentFamily) {	// if they have different queue families we will have to use concurrent mode
			createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
			createInfo.queueFamilyIndexCount = 2;	// We set up that there will be 2 concurrent queues
			createInfo.pQueueFamilyIndices = queueFamilyIndices; // we give the indices of the queues
		}
		else {	// if the share family queue we can use exclusive mode where only one queue has acces at the same time to an image which usually improves performance
			createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
		}

		createInfo.preTransform = swapChainSupport.capabilities.currentTransform;	// We don't want to set any transformations to the images in the swap chain so we let them as they are
		createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;	// specify the desired alpha channel for blending
		createInfo.presentMode = presentMode;	// set the present mode
		createInfo.clipped = VK_TRUE;	// we allow the system to remove anything that has been clipped

		if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {	// We finally create the swap chain with the information of the struct and throw a
																							// runtime error if it fails
			throw std::runtime_error("failed to create swap chain!");
		}

		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);	// get the number of images
		swapChainImages.resize(imageCount);	// resize the number of images in the struct to the actual number of images
		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());	// fill in the data of the images in the array

		swapChainImageFormat = surfaceFormat.format;	// set the format for the member variable
		swapChainExtent = extent;	// set the format for the member variable
	}

	// Creates an image view for every image of the swap chain
	void createImageViews() {
		swapChainImageViews.resize(swapChainImages.size());

		for (size_t i = 0; i < swapChainImages.size(); i++) {
			swapChainImageViews[i] = createImageView(swapChainImages[i], swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT);
		}
	}

	// Create a new render pass for the gui
	void createRenderPassGUI() {
		VkAttachmentDescription attachment = {};	// single color buffer 
		attachment.format = swapChainImageFormat;	// we use the same format as in the swap chain
		attachment.samples = VK_SAMPLE_COUNT_1_BIT;	// we don't do multisampling right now so, we have only 1 sample
		attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;	// The data in the attachment before rendering will be set to a constant value
		attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;	// The data in the attachment after rendering in this case will be stored and can later on be readed from the buffer
		attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;	// As we don't use the stencil buffer we have undefined contents in that buffer
		attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;	// The same with this one, after rendering the contents will be undefined as we do not worry about them
		attachment.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;	// specifies how should be the layout before the render pass, in this case is undefined
		attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;	// specifies the layout to automatically transition in this case by presenting the image to the swap chain

		VkAttachmentReference colorAttachmentRef = {};
		colorAttachmentRef.attachment = 0;	// index of the reference attachment
		colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;	// specifies the desired layout during supasses

		// A subpass is a subsequent rendering pass using information of previous subpasses 
		VkSubpassDescription subpass = {};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;	// Specifies the type of subpass that we will do in our case graphics
		subpass.colorAttachmentCount = 1;	// is the number of color attachments
		subpass.pColorAttachments = &colorAttachmentRef;	// pointer to the color attachment ref that will be used in that subpass

		// The transition between subpasses are controlled by the dependency and we will have to configure when should the start transition should start
		VkSubpassDependency dependency = {};
		dependency.srcSubpass = VK_SUBPASS_EXTERNAL;	// is the index of the first subpass in the dependency
		dependency.dstSubpass = 0;	// is the index of the second subpass in the dependency
		dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;	// Specifies when should the first subpass should start (in which stage)
		dependency.srcAccessMask = 0;	// Specifies the source access 
		dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;	// Specifies until which stage it should wait 
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;	// Specifies the destination access mask

		std::array<VkAttachmentDescription, 1> attachments = { attachment };
		// Has all the data to be able to construct the render pass
		VkRenderPassCreateInfo renderPassInfo = {};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;	// set the type of the struct
		renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());	// number of attachments in this render pass
		renderPassInfo.pAttachments = attachments.data();	// pointer to the color attachment buffer
		renderPassInfo.subpassCount = 1;	// is the number of subpasses that will be created
		renderPassInfo.pSubpasses = &subpass;	// pointer to the subpass that will be used
		renderPassInfo.dependencyCount = 1;	// is the number of memory dependencies between subpasses
		renderPassInfo.pDependencies = &dependency;	// pointer to the dependencies struct

		if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPassGUI) != VK_SUCCESS) {	// Creates the render pass with the data of the struct and 
																								// throw a runtime error if it fails
			throw std::runtime_error("failed to create render pass!");
		}
	}

	// Configures how to configurate the color buffer and how could the render pass and subpasses be implemented
	void createRenderPass() {
		VkAttachmentDescription colorAttachment = {};	// single color buffer 
		colorAttachment.format = swapChainImageFormat;	// we use the same format as in the swap chain
		colorAttachment.samples = msaaSamples;	// we don't do multisampling right now so, we have only 1 sample
		colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;	// The data in the attachment before rendering will be set to a constant value
		colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;	// The data in the attachment after rendering in this case will be stored and can later on be readed from the buffer
		colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;	// As we don't use the stencil buffer we have undefined contents in that buffer
		colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;	// The same with this one, after rendering the contents will be undefined as we do not worry about them
		colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;	// specifies how should be the layout before the render pass, in this case is undefined
		colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;	// specifies the layout to automatically transition in this case by presenting the image to the swap chain
		

		VkAttachmentDescription depthAttachment = {};	// Set up all the information for the depth buffer to be able to work
		depthAttachment.format = findDepthFormat();
		depthAttachment.samples = msaaSamples;
		depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkAttachmentDescription colorAttachmentResolve = {};
		colorAttachmentResolve.format = swapChainImageFormat;
		colorAttachmentResolve.samples = VK_SAMPLE_COUNT_1_BIT;
		colorAttachmentResolve.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachmentResolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		colorAttachmentResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachmentResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		colorAttachmentResolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		colorAttachmentResolve.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentReference colorAttachmentRef = {};
		colorAttachmentRef.attachment = 0;	// index of the reference attachment
		colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;	// specifies the desired layout during supasses

		VkAttachmentReference depthAttachmentRef = {};
		depthAttachmentRef.attachment = 1;
		depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkAttachmentReference colorAttachmentResolveRef{};
		colorAttachmentResolveRef.attachment = 2;
		colorAttachmentResolveRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
		
		// A subpass is a subsequent rendering pass using information of previous subpasses 
		VkSubpassDescription subpass = {};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;	// Specifies the type of subpass that we will do in our case graphics
		subpass.colorAttachmentCount = 1;	// is the number of color attachments
		subpass.pColorAttachments = &colorAttachmentRef;	// pointer to the color attachment ref that will be used in that subpass
		subpass.pDepthStencilAttachment = &depthAttachmentRef;
		if (msaaSamples != VK_SAMPLE_COUNT_1_BIT)
		{
			subpass.pResolveAttachments = &colorAttachmentResolveRef;
		}

		// The transition between subpasses are controlled by the dependency and we will have to configure when should the start transition should start
		VkSubpassDependency dependency = {};
		dependency.srcSubpass = VK_SUBPASS_EXTERNAL;	// is the index of the first subpass in the dependency
		dependency.dstSubpass = 0;	// is the index of the second subpass in the dependency
		dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;	// Specifies when should the first subpass should start (in which stage)
		dependency.srcAccessMask = 0;	// Specifies the source access 
		dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;	// Specifies until which stage it should wait 
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;	// Specifies the destination access mask

		VkRenderPassCreateInfo renderPassInfo = {};
		if (msaaSamples != VK_SAMPLE_COUNT_1_BIT)
		{
			std::array<VkAttachmentDescription, 3> attachments = { colorAttachment, depthAttachment, colorAttachmentResolve };
			renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;	// set the type of the struct
			renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());	// number of attachments in this render pass
			renderPassInfo.pAttachments = attachments.data();	// pointer to the color attachment buffer
			renderPassInfo.subpassCount = 1;	// is the number of subpasses that will be created
			renderPassInfo.pSubpasses = &subpass;	// pointer to the subpass that will be used
			renderPassInfo.dependencyCount = 1;	// is the number of memory dependencies between subpasses
			renderPassInfo.pDependencies = &dependency;	// pointer to the dependencies struct
		}
		else
		{
			std::array<VkAttachmentDescription, 2> attachments = { colorAttachment, depthAttachment };
			renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;	// set the type of the struct
			renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());	// number of attachments in this render pass
			renderPassInfo.pAttachments = attachments.data();	// pointer to the color attachment buffer
			renderPassInfo.subpassCount = 1;	// is the number of subpasses that will be created
			renderPassInfo.pSubpasses = &subpass;	// pointer to the subpass that will be used
			renderPassInfo.dependencyCount = 1;	// is the number of memory dependencies between subpasses
			renderPassInfo.pDependencies = &dependency;	// pointer to the dependencies struct
		}

		if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {	// Creates the render pass with the data of the struct and 
																								// throw a runtime error if it fails
			throw std::runtime_error("failed to create render pass!");
		}
	}

	// Describe the layout that each of the uniform buffers will have
	void createDescriptorSetLayout() {
		VkDescriptorSetLayoutBinding uboLayoutBinding = {};
		uboLayoutBinding.binding = 0;
		uboLayoutBinding.descriptorCount = 1;
		uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		uboLayoutBinding.pImmutableSamplers = nullptr;
		uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

		VkDescriptorSetLayoutBinding samplerLayoutBinding = {};	// this descriptors allows the shaders to access an image resource
		samplerLayoutBinding.binding = 1;
		samplerLayoutBinding.descriptorCount = 1;
		samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		samplerLayoutBinding.pImmutableSamplers = nullptr;
		samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutBinding lboLayoutBinding = {};
		lboLayoutBinding.binding = 2;
		lboLayoutBinding.descriptorCount = 1;
		lboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		lboLayoutBinding.pImmutableSamplers = nullptr;
		lboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

		VkDescriptorSetLayoutBinding gaussianNoiseLayoutBinding = {};
		gaussianNoiseLayoutBinding.binding = 3;
		gaussianNoiseLayoutBinding.descriptorCount = 1;
		gaussianNoiseLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		gaussianNoiseLayoutBinding.pImmutableSamplers = nullptr;
		gaussianNoiseLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		VkDescriptorSetLayoutBinding computeShaderLayoutBindingHk0 = {};	// this descriptors allows the shaders to access an image resource
		computeShaderLayoutBindingHk0.binding = 4;
		computeShaderLayoutBindingHk0.descriptorCount = 1;
		computeShaderLayoutBindingHk0.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		computeShaderLayoutBindingHk0.pImmutableSamplers = nullptr;
		computeShaderLayoutBindingHk0.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		VkDescriptorSetLayoutBinding computeShaderFinalLayoutBindingHk0 = {};	// this descriptors allows the shaders to access an image resource
		computeShaderFinalLayoutBindingHk0.binding = 5;
		computeShaderFinalLayoutBindingHk0.descriptorCount = 1;
		computeShaderFinalLayoutBindingHk0.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		computeShaderFinalLayoutBindingHk0.pImmutableSamplers = nullptr;
		computeShaderFinalLayoutBindingHk0.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutBinding computeShaderLayoutBindingHk0minus = {};	// this descriptors allows the shaders to access an image resource
		computeShaderLayoutBindingHk0minus.binding = 6;
		computeShaderLayoutBindingHk0minus.descriptorCount = 1;
		computeShaderLayoutBindingHk0minus.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		computeShaderLayoutBindingHk0minus.pImmutableSamplers = nullptr;
		computeShaderLayoutBindingHk0minus.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		VkDescriptorSetLayoutBinding computeShaderFinalLayoutBindingHk0minus = {};	// this descriptors allows the shaders to access an image resource
		computeShaderFinalLayoutBindingHk0minus.binding = 7;
		computeShaderFinalLayoutBindingHk0minus.descriptorCount = 1;
		computeShaderFinalLayoutBindingHk0minus.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		computeShaderFinalLayoutBindingHk0minus.pImmutableSamplers = nullptr;
		computeShaderFinalLayoutBindingHk0minus.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutBinding computeShaderLayoutBindingHkt = {};	// this descriptors allows the shaders to access an image resource
		computeShaderLayoutBindingHkt.binding = 8;
		computeShaderLayoutBindingHkt.descriptorCount = 1;
		computeShaderLayoutBindingHkt.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		computeShaderLayoutBindingHkt.pImmutableSamplers = nullptr;
		computeShaderLayoutBindingHkt.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		VkDescriptorSetLayoutBinding computeShaderFinalLayoutBindingHkt = {};	// this descriptors allows the shaders to access an image resource
		computeShaderFinalLayoutBindingHkt.binding = 9;
		computeShaderFinalLayoutBindingHkt.descriptorCount = 1;
		computeShaderFinalLayoutBindingHkt.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		computeShaderFinalLayoutBindingHkt.pImmutableSamplers = nullptr;
		computeShaderFinalLayoutBindingHkt.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
		
		VkDescriptorSetLayoutBinding hktLayoutBinding = {};
		hktLayoutBinding.binding = 10;
		hktLayoutBinding.descriptorCount = 1;
		hktLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		hktLayoutBinding.pImmutableSamplers = nullptr;
		hktLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		VkDescriptorSetLayoutBinding computeShaderLayoutBindingFFTAux = {};	// this descriptors allows the shaders to access an image resource
		computeShaderLayoutBindingFFTAux.binding = 11;
		computeShaderLayoutBindingFFTAux.descriptorCount = 1;
		computeShaderLayoutBindingFFTAux.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		computeShaderLayoutBindingFFTAux.pImmutableSamplers = nullptr;
		computeShaderLayoutBindingFFTAux.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		VkDescriptorSetLayoutBinding computeShaderFinalLayoutBindingFFTAux = {};	// this descriptors allows the shaders to access an image resource
		computeShaderFinalLayoutBindingFFTAux.binding = 12;
		computeShaderFinalLayoutBindingFFTAux.descriptorCount = 1;
		computeShaderFinalLayoutBindingFFTAux.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		computeShaderFinalLayoutBindingFFTAux.pImmutableSamplers = nullptr;
		computeShaderFinalLayoutBindingFFTAux.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutBinding FFTAuxLayoutBinding = {};
		FFTAuxLayoutBinding.binding = 13;
		FFTAuxLayoutBinding.descriptorCount = 1;
		FFTAuxLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		FFTAuxLayoutBinding.pImmutableSamplers = nullptr;
		FFTAuxLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		VkDescriptorSetLayoutBinding computeShaderLayoutBindingFFTAlternate = {};	// this descriptors allows the shaders to access an image resource
		computeShaderLayoutBindingFFTAlternate.binding = 14;
		computeShaderLayoutBindingFFTAlternate.descriptorCount = 1;
		computeShaderLayoutBindingFFTAlternate.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		computeShaderLayoutBindingFFTAlternate.pImmutableSamplers = nullptr;
		computeShaderLayoutBindingFFTAlternate.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		VkDescriptorSetLayoutBinding computeShaderFinalLayoutBindingFFTAlternate = {};	// this descriptors allows the shaders to access an image resource
		computeShaderFinalLayoutBindingFFTAlternate.binding = 15;
		computeShaderFinalLayoutBindingFFTAlternate.descriptorCount = 1;
		computeShaderFinalLayoutBindingFFTAlternate.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		computeShaderFinalLayoutBindingFFTAlternate.pImmutableSamplers = nullptr;
		computeShaderFinalLayoutBindingFFTAlternate.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutBinding computeShaderLayoutBindingHeightMap = {};	// this descriptors allows the shaders to access an image resource
		computeShaderLayoutBindingHeightMap.binding = 16;
		computeShaderLayoutBindingHeightMap.descriptorCount = 1;
		computeShaderLayoutBindingHeightMap.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		computeShaderLayoutBindingHeightMap.pImmutableSamplers = nullptr;
		computeShaderLayoutBindingHeightMap.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		VkDescriptorSetLayoutBinding computeShaderFinalLayoutBindingHeightMap = {};	// this descriptors allows the shaders to access an image resource
		computeShaderFinalLayoutBindingHeightMap.binding = 17;
		computeShaderFinalLayoutBindingHeightMap.descriptorCount = 1;
		computeShaderFinalLayoutBindingHeightMap.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		computeShaderFinalLayoutBindingHeightMap.pImmutableSamplers = nullptr;
		computeShaderFinalLayoutBindingHeightMap.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;


		VkDescriptorSetLayoutBinding computeShaderFinalLayoutBindingHeightMapVertex = {};	// this descriptors allows the shaders to access an image resource
		computeShaderFinalLayoutBindingHeightMapVertex.binding = 18;
		computeShaderFinalLayoutBindingHeightMapVertex.descriptorCount = 1;
		computeShaderFinalLayoutBindingHeightMapVertex.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		computeShaderFinalLayoutBindingHeightMapVertex.pImmutableSamplers = nullptr;
		computeShaderFinalLayoutBindingHeightMapVertex.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

		VkDescriptorSetLayoutBinding computeShaderLayoutBindingSlopeX = {};	// this descriptors allows the shaders to access an image resource
		computeShaderLayoutBindingSlopeX.binding = 19;
		computeShaderLayoutBindingSlopeX.descriptorCount = 1;
		computeShaderLayoutBindingSlopeX.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		computeShaderLayoutBindingSlopeX.pImmutableSamplers = nullptr;
		computeShaderLayoutBindingSlopeX.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		VkDescriptorSetLayoutBinding computeShaderFinalLayoutBindingSlopeX = {};	// this descriptors allows the shaders to access an image resource
		computeShaderFinalLayoutBindingSlopeX.binding = 20;
		computeShaderFinalLayoutBindingSlopeX.descriptorCount = 1;
		computeShaderFinalLayoutBindingSlopeX.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		computeShaderFinalLayoutBindingSlopeX.pImmutableSamplers = nullptr;
		computeShaderFinalLayoutBindingSlopeX.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutBinding computeShaderLayoutBindingSlopeXAlternate = {};	// this descriptors allows the shaders to access an image resource
		computeShaderLayoutBindingSlopeXAlternate.binding = 21;
		computeShaderLayoutBindingSlopeXAlternate.descriptorCount = 1;
		computeShaderLayoutBindingSlopeXAlternate.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		computeShaderLayoutBindingSlopeXAlternate.pImmutableSamplers = nullptr;
		computeShaderLayoutBindingSlopeXAlternate.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		VkDescriptorSetLayoutBinding computeShaderFinalLayoutBindingSlopeXAlternate = {};	// this descriptors allows the shaders to access an image resource
		computeShaderFinalLayoutBindingSlopeXAlternate.binding = 22;
		computeShaderFinalLayoutBindingSlopeXAlternate.descriptorCount = 1;
		computeShaderFinalLayoutBindingSlopeXAlternate.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		computeShaderFinalLayoutBindingSlopeXAlternate.pImmutableSamplers = nullptr;
		computeShaderFinalLayoutBindingSlopeXAlternate.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutBinding computeShaderLayoutBindingSlopeZ = {};	// this descriptors allows the shaders to access an image resource
		computeShaderLayoutBindingSlopeZ.binding = 23;
		computeShaderLayoutBindingSlopeZ.descriptorCount = 1;
		computeShaderLayoutBindingSlopeZ.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		computeShaderLayoutBindingSlopeZ.pImmutableSamplers = nullptr;
		computeShaderLayoutBindingSlopeZ.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		VkDescriptorSetLayoutBinding computeShaderFinalLayoutBindingSlopeZ = {};	// this descriptors allows the shaders to access an image resource
		computeShaderFinalLayoutBindingSlopeZ.binding = 24;
		computeShaderFinalLayoutBindingSlopeZ.descriptorCount = 1;
		computeShaderFinalLayoutBindingSlopeZ.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		computeShaderFinalLayoutBindingSlopeZ.pImmutableSamplers = nullptr;
		computeShaderFinalLayoutBindingSlopeZ.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutBinding computeShaderLayoutBindingSlopeZAlternate = {};	// this descriptors allows the shaders to access an image resource
		computeShaderLayoutBindingSlopeZAlternate.binding = 25;
		computeShaderLayoutBindingSlopeZAlternate.descriptorCount = 1;
		computeShaderLayoutBindingSlopeZAlternate.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		computeShaderLayoutBindingSlopeZAlternate.pImmutableSamplers = nullptr;
		computeShaderLayoutBindingSlopeZAlternate.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		VkDescriptorSetLayoutBinding computeShaderFinalLayoutBindingSlopeZAlternate = {};	// this descriptors allows the shaders to access an image resource
		computeShaderFinalLayoutBindingSlopeZAlternate.binding = 26;
		computeShaderFinalLayoutBindingSlopeZAlternate.descriptorCount = 1;
		computeShaderFinalLayoutBindingSlopeZAlternate.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		computeShaderFinalLayoutBindingSlopeZAlternate.pImmutableSamplers = nullptr;
		computeShaderFinalLayoutBindingSlopeZAlternate.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutBinding computeShaderLayoutBindingDispX = {};	// this descriptors allows the shaders to access an image resource
		computeShaderLayoutBindingDispX.binding = 27;
		computeShaderLayoutBindingDispX.descriptorCount = 1;
		computeShaderLayoutBindingDispX.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		computeShaderLayoutBindingDispX.pImmutableSamplers = nullptr;
		computeShaderLayoutBindingDispX.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		VkDescriptorSetLayoutBinding computeShaderFinalLayoutBindingDispX = {};	// this descriptors allows the shaders to access an image resource
		computeShaderFinalLayoutBindingDispX.binding = 28;
		computeShaderFinalLayoutBindingDispX.descriptorCount = 1;
		computeShaderFinalLayoutBindingDispX.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		computeShaderFinalLayoutBindingDispX.pImmutableSamplers = nullptr;
		computeShaderFinalLayoutBindingDispX.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutBinding computeShaderLayoutBindingDispXAlternate = {};	// this descriptors allows the shaders to access an image resource
		computeShaderLayoutBindingDispXAlternate.binding = 29;
		computeShaderLayoutBindingDispXAlternate.descriptorCount = 1;
		computeShaderLayoutBindingDispXAlternate.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		computeShaderLayoutBindingDispXAlternate.pImmutableSamplers = nullptr;
		computeShaderLayoutBindingDispXAlternate.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		VkDescriptorSetLayoutBinding computeShaderFinalLayoutBindingDispXAlternate = {};	// this descriptors allows the shaders to access an image resource
		computeShaderFinalLayoutBindingDispXAlternate.binding = 30;
		computeShaderFinalLayoutBindingDispXAlternate.descriptorCount = 1;
		computeShaderFinalLayoutBindingDispXAlternate.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		computeShaderFinalLayoutBindingDispXAlternate.pImmutableSamplers = nullptr;
		computeShaderFinalLayoutBindingDispXAlternate.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutBinding computeShaderLayoutBindingDispZ = {};	// this descriptors allows the shaders to access an image resource
		computeShaderLayoutBindingDispZ.binding = 31;
		computeShaderLayoutBindingDispZ.descriptorCount = 1;
		computeShaderLayoutBindingDispZ.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		computeShaderLayoutBindingDispZ.pImmutableSamplers = nullptr;
		computeShaderLayoutBindingDispZ.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		VkDescriptorSetLayoutBinding computeShaderFinalLayoutBindingDispZ = {};	// this descriptors allows the shaders to access an image resource
		computeShaderFinalLayoutBindingDispZ.binding = 32;
		computeShaderFinalLayoutBindingDispZ.descriptorCount = 1;
		computeShaderFinalLayoutBindingDispZ.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		computeShaderFinalLayoutBindingDispZ.pImmutableSamplers = nullptr;
		computeShaderFinalLayoutBindingDispZ.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutBinding computeShaderLayoutBindingDispZAlternate = {};	// this descriptors allows the shaders to access an image resource
		computeShaderLayoutBindingDispZAlternate.binding = 33;
		computeShaderLayoutBindingDispZAlternate.descriptorCount = 1;
		computeShaderLayoutBindingDispZAlternate.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		computeShaderLayoutBindingDispZAlternate.pImmutableSamplers = nullptr;
		computeShaderLayoutBindingDispZAlternate.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		VkDescriptorSetLayoutBinding computeShaderFinalLayoutBindingDispZAlternate = {};	// this descriptors allows the shaders to access an image resource
		computeShaderFinalLayoutBindingDispZAlternate.binding = 34;
		computeShaderFinalLayoutBindingDispZAlternate.descriptorCount = 1;
		computeShaderFinalLayoutBindingDispZAlternate.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		computeShaderFinalLayoutBindingDispZAlternate.pImmutableSamplers = nullptr;
		computeShaderFinalLayoutBindingDispZAlternate.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutBinding computeShaderLayoutBindingSlopeXFinal = {};	// this descriptors allows the shaders to access an image resource
		computeShaderLayoutBindingSlopeXFinal.binding = 35;
		computeShaderLayoutBindingSlopeXFinal.descriptorCount = 1;
		computeShaderLayoutBindingSlopeXFinal.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		computeShaderLayoutBindingSlopeXFinal.pImmutableSamplers = nullptr;
		computeShaderLayoutBindingSlopeXFinal.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		VkDescriptorSetLayoutBinding computeShaderFinalLayoutBindingSlopeXFinal = {};	// this descriptors allows the shaders to access an image resource
		computeShaderFinalLayoutBindingSlopeXFinal.binding = 36;
		computeShaderFinalLayoutBindingSlopeXFinal.descriptorCount = 1;
		computeShaderFinalLayoutBindingSlopeXFinal.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		computeShaderFinalLayoutBindingSlopeXFinal.pImmutableSamplers = nullptr;
		computeShaderFinalLayoutBindingSlopeXFinal.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;


		VkDescriptorSetLayoutBinding computeShaderFinalLayoutBindingSlopeXVertex = {};	// this descriptors allows the shaders to access an image resource
		computeShaderFinalLayoutBindingSlopeXVertex.binding = 37;
		computeShaderFinalLayoutBindingSlopeXVertex.descriptorCount = 1;
		computeShaderFinalLayoutBindingSlopeXVertex.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		computeShaderFinalLayoutBindingSlopeXVertex.pImmutableSamplers = nullptr;
		computeShaderFinalLayoutBindingSlopeXVertex.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

		VkDescriptorSetLayoutBinding computeShaderLayoutBindingSlopeZFinal = {};	// this descriptors allows the shaders to access an image resource
		computeShaderLayoutBindingSlopeZFinal.binding = 38;
		computeShaderLayoutBindingSlopeZFinal.descriptorCount = 1;
		computeShaderLayoutBindingSlopeZFinal.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		computeShaderLayoutBindingSlopeZFinal.pImmutableSamplers = nullptr;
		computeShaderLayoutBindingSlopeZFinal.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		VkDescriptorSetLayoutBinding computeShaderFinalLayoutBindingSlopeZFinal = {};	// this descriptors allows the shaders to access an image resource
		computeShaderFinalLayoutBindingSlopeZFinal.binding = 39;
		computeShaderFinalLayoutBindingSlopeZFinal.descriptorCount = 1;
		computeShaderFinalLayoutBindingSlopeZFinal.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		computeShaderFinalLayoutBindingSlopeZFinal.pImmutableSamplers = nullptr;
		computeShaderFinalLayoutBindingSlopeZFinal.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;


		VkDescriptorSetLayoutBinding computeShaderFinalLayoutBindingSlopeZVertex = {};	// this descriptors allows the shaders to access an image resource
		computeShaderFinalLayoutBindingSlopeZVertex.binding = 40;
		computeShaderFinalLayoutBindingSlopeZVertex.descriptorCount = 1;
		computeShaderFinalLayoutBindingSlopeZVertex.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		computeShaderFinalLayoutBindingSlopeZVertex.pImmutableSamplers = nullptr;
		computeShaderFinalLayoutBindingSlopeZVertex.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

		VkDescriptorSetLayoutBinding computeShaderLayoutBindingDispXFinal = {};	// this descriptors allows the shaders to access an image resource
		computeShaderLayoutBindingDispXFinal.binding = 41;
		computeShaderLayoutBindingDispXFinal.descriptorCount = 1;
		computeShaderLayoutBindingDispXFinal.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		computeShaderLayoutBindingDispXFinal.pImmutableSamplers = nullptr;
		computeShaderLayoutBindingDispXFinal.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		VkDescriptorSetLayoutBinding computeShaderFinalLayoutBindingDispXFinal = {};	// this descriptors allows the shaders to access an image resource
		computeShaderFinalLayoutBindingDispXFinal.binding = 42;
		computeShaderFinalLayoutBindingDispXFinal.descriptorCount = 1;
		computeShaderFinalLayoutBindingDispXFinal.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		computeShaderFinalLayoutBindingDispXFinal.pImmutableSamplers = nullptr;
		computeShaderFinalLayoutBindingDispXFinal.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;


		VkDescriptorSetLayoutBinding computeShaderFinalLayoutBindingDispXVertex = {};	// this descriptors allows the shaders to access an image resource
		computeShaderFinalLayoutBindingDispXVertex.binding = 43;
		computeShaderFinalLayoutBindingDispXVertex.descriptorCount = 1;
		computeShaderFinalLayoutBindingDispXVertex.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		computeShaderFinalLayoutBindingDispXVertex.pImmutableSamplers = nullptr;
		computeShaderFinalLayoutBindingDispXVertex.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

		VkDescriptorSetLayoutBinding computeShaderLayoutBindingDispZFinal = {};	// this descriptors allows the shaders to access an image resource
		computeShaderLayoutBindingDispZFinal.binding = 44;
		computeShaderLayoutBindingDispZFinal.descriptorCount = 1;
		computeShaderLayoutBindingDispZFinal.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		computeShaderLayoutBindingDispZFinal.pImmutableSamplers = nullptr;
		computeShaderLayoutBindingDispZFinal.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		VkDescriptorSetLayoutBinding computeShaderFinalLayoutBindingDispZFinal = {};	// this descriptors allows the shaders to access an image resource
		computeShaderFinalLayoutBindingDispZFinal.binding = 45;
		computeShaderFinalLayoutBindingDispZFinal.descriptorCount = 1;
		computeShaderFinalLayoutBindingDispZFinal.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		computeShaderFinalLayoutBindingDispZFinal.pImmutableSamplers = nullptr;
		computeShaderFinalLayoutBindingDispZFinal.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;


		VkDescriptorSetLayoutBinding computeShaderFinalLayoutBindingDispZVertex = {};	// this descriptors allows the shaders to access an image resource
		computeShaderFinalLayoutBindingDispZVertex.binding = 46;
		computeShaderFinalLayoutBindingDispZVertex.descriptorCount = 1;
		computeShaderFinalLayoutBindingDispZVertex.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		computeShaderFinalLayoutBindingDispZVertex.pImmutableSamplers = nullptr;
		computeShaderFinalLayoutBindingDispZVertex.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;


		std::array<VkDescriptorSetLayoutBinding, 47> bindings = 
			{ uboLayoutBinding, samplerLayoutBinding, lboLayoutBinding,
			gaussianNoiseLayoutBinding, 
			computeShaderLayoutBindingHk0, computeShaderFinalLayoutBindingHk0, 
			computeShaderLayoutBindingHk0minus, computeShaderFinalLayoutBindingHk0minus,
			computeShaderLayoutBindingHkt, computeShaderFinalLayoutBindingHkt, hktLayoutBinding,
			computeShaderLayoutBindingFFTAux , computeShaderFinalLayoutBindingFFTAux, FFTAuxLayoutBinding,
			computeShaderLayoutBindingFFTAlternate , computeShaderFinalLayoutBindingFFTAlternate,
			computeShaderLayoutBindingHeightMap , computeShaderFinalLayoutBindingHeightMap, computeShaderFinalLayoutBindingHeightMapVertex,
			computeShaderLayoutBindingSlopeX , computeShaderFinalLayoutBindingSlopeX,
			computeShaderLayoutBindingSlopeXAlternate , computeShaderFinalLayoutBindingSlopeXAlternate,
			computeShaderLayoutBindingSlopeZ , computeShaderFinalLayoutBindingSlopeZ,
			computeShaderLayoutBindingSlopeZAlternate , computeShaderFinalLayoutBindingSlopeZAlternate,
			computeShaderLayoutBindingDispX , computeShaderFinalLayoutBindingDispX,
			computeShaderLayoutBindingDispXAlternate , computeShaderFinalLayoutBindingDispXAlternate,
			computeShaderLayoutBindingDispZ , computeShaderFinalLayoutBindingDispZ, 
			computeShaderLayoutBindingDispZAlternate, computeShaderFinalLayoutBindingDispZAlternate,
			computeShaderLayoutBindingSlopeXFinal, computeShaderFinalLayoutBindingSlopeXFinal, computeShaderFinalLayoutBindingSlopeXVertex,
			computeShaderLayoutBindingSlopeZFinal, computeShaderFinalLayoutBindingSlopeZFinal, computeShaderFinalLayoutBindingSlopeZVertex,
			computeShaderLayoutBindingDispXFinal, computeShaderFinalLayoutBindingDispXFinal, computeShaderFinalLayoutBindingDispXVertex,
			computeShaderLayoutBindingDispZFinal, computeShaderFinalLayoutBindingDispZFinal, computeShaderFinalLayoutBindingDispZVertex };

		VkDescriptorSetLayoutCreateInfo layoutInfo = {};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
		layoutInfo.pBindings = bindings.data();

		if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create descriptor set layout!");
		}
	}

	// This method will create all the stages of the graphics pipeline, it will load those stages that are programmable and configure those that are configurable
	void createGraphicsPipeline() {
		auto vertShaderCode = readFile("vert.spv");	// Read the already compiled vertex shader file
		auto fragShaderCode = readFile("frag.spv"); // Read the already compiled fragment shader file

		VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);	// Wrap the bytecode of the vertex shader code into a VkShaderModule object
		VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);	// Wrap the bytecode of the fragment shader code into a VkShaderModule object

		VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;	// set the type of the struct
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;	// specify which stage of the pipeline is defining
		vertShaderStageInfo.module = vertShaderModule;	// Contains the shader object
		vertShaderStageInfo.pName = "main";	// specify the entry point of the shader

		VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
		fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;	// set the type of the struct
		fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;	// specify which stage of the pipeline is defining
		fragShaderStageInfo.module = fragShaderModule;	// Contains the shader object
		fragShaderStageInfo.pName = "main";	// specify the entry point of the shader

		VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };	// Holds all the stages of the pipeline that will be implemented

		auto bindingDescription = Vertex::getBindingDescription();
		auto attributeDescriptions = Vertex::getAttributeDescriptions();

		// It specifies the format in which the data will be pass down to the vertex shader 
		VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;	// set the type of the struct
		vertexInputInfo.vertexBindingDescriptionCount = 1;	// is the number of vertex binding descriptors provided 
		vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());	// is the number of vertex attribute descriptions provided
		vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
		vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

		// It specifies which type of primitive will be drawn and if primitives should restart
		VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;	// set the type of the struct
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;	// the primitive type will be triangles specified by 3 vertices without reuse 
		inputAssembly.primitiveRestartEnable = VK_FALSE;	// disable the restart of primitives types

		// defines a transformation from the image to the framebuffer
		VkViewport viewport = {};
		viewport.x = 0.0f;	// Viewport upper left coordinate
		viewport.y = 0.0f;
		viewport.width = (float)swapChainExtent.width;	// Viewport width
		viewport.height = (float)swapChainExtent.height;	// Viewport height
		viewport.minDepth = 0.0f;	// is the depth range of the viewport where the min value can be greater than the max value. They have to be between 0 and 1
		viewport.maxDepth = 1.0f;

		// defines in which regions pixels will actually be stored
		VkRect2D scissor = {};
		scissor.offset = { 0, 0 };	// specifies the rectangle offset
		scissor.extent = swapChainExtent;	// specifies the rectangle extent using the same extent that in the swap chain

		// Struct to combine the viewport with the scissor 
		VkPipelineViewportStateCreateInfo viewportState = {};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;	// set the type of the struct
		viewportState.viewportCount = 1;	// specifies the number of viewports used by the pipeline
		viewportState.pViewports = &viewport;	// pointer to the viewport that will be used
		viewportState.scissorCount = 1;	// specifies the number of scissors used by the pipeline
		viewportState.pScissors = &scissor;	// pointer to the scissor that will be used

		// Turns the geometry shaped by the vertices and transform it into fragments to later be coloured by the fragment shader
		// Also performs depth test, face culling and the scissor test
		VkPipelineRasterizationStateCreateInfo rasterizer = {};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;	// set the type of the struct
		rasterizer.depthClampEnable = VK_FALSE;	// If it's false it will discard the fragments beyond the near and far plane, if it's true it will clamp them to the limits
		rasterizer.rasterizerDiscardEnable = VK_FALSE;	// If it's true it discard all fragments and doesn't give any output
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;	// there are 3 modes, this one fill in the area that the polygon makes.
		rasterizer.lineWidth = 1.0f;	// Describes the thickness of the lines
		rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;	// determines the type of culling that will be applied 
		rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;	// specifies the vertex order for the faces
		rasterizer.depthBiasEnable = VK_FALSE;	// enables or disable applying a depth bias 

		// Is a way to avoid aliasing, it requires to enable a GPU feature 
		VkPipelineMultisampleStateCreateInfo multisampling = {};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;	// set the type of the struct
		multisampling.sampleShadingEnable = VK_FALSE;	// enables or disable sample shading
		multisampling.rasterizationSamples = msaaSamples;

		VkPipelineDepthStencilStateCreateInfo depthStencil = {};
		depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthStencil.depthTestEnable = VK_TRUE;
		depthStencil.depthWriteEnable = VK_TRUE;
		depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
		depthStencil.depthBoundsTestEnable = VK_FALSE;
		depthStencil.stencilTestEnable = VK_FALSE;

		// proccess the combination of colors from the fragment shader and the color that was previously in the buffer
		// it does it by attached framebuffer
		VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
		// specifies which channels are open to be modified bitwise
		colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_TRUE;	// if it's set to false it will just use the color out of the fragments shader otherwise it will perform color blending
		colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD; 
		colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
		// perform color blending, but it is set globally
		VkPipelineColorBlendStateCreateInfo colorBlending = {};
		colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;	// set the type of the struct
		colorBlending.logicOpEnable = VK_FALSE;	// Controls wether to apply Logical Operations
		colorBlending.logicOp = VK_LOGIC_OP_COPY;	// Selects the logical operation to apply
		colorBlending.attachmentCount = 1;	// Defines the number of VkPipelineColorBlendAttachmentState objects 
		colorBlending.pAttachments = &colorBlendAttachment;	// pointer to the attachments
		colorBlending.blendConstants[0] = 0.0f;	// constant values that are use while blending for R, G, B, A
		colorBlending.blendConstants[1] = 0.0f;
		colorBlending.blendConstants[2] = 0.0f;
		colorBlending.blendConstants[3] = 0.0f;

		// Here it is specified which uniform variables will be used in the shaders to pass information
		VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;	// set the type of the struct
		pipelineLayoutInfo.setLayoutCount = 1;	// is the number of descriptors for the layout
		pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

		VkPushConstantRange pushConstantRange = {};
		pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
		pushConstantRange.offset = 0;
		pushConstantRange.size = sizeof(pushConstantsRenderScenePipeline);

		pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
		pipelineLayoutInfo.pushConstantRangeCount = 1;

		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {	// Creates the layout of the pipeline with the data of the struct
																											// Throws a runtime error if it fails
			throw std::runtime_error("failed to create pipeline layout!");
		}

		VkGraphicsPipelineCreateInfo pipelineInfo = {};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;	// set the type of the struct
		pipelineInfo.stageCount = 2;	// is the number of stages 
		pipelineInfo.pStages = shaderStages;	// is a pointer to the array of the vertex and fragment shader
		pipelineInfo.pVertexInputState = &vertexInputInfo;	// is a pointer to the struct for the vertex input stage
		pipelineInfo.pInputAssemblyState = &inputAssembly;	// is a pointer to the struct for the input assembly
		pipelineInfo.pViewportState = &viewportState;	// is a pointer to the struct for the viewport
		pipelineInfo.pRasterizationState = &rasterizer;	// is a pointer to the struct for the rasterization stage
		pipelineInfo.pMultisampleState = &multisampling;	// is a pointer to the struct for multisampling
		pipelineInfo.pDepthStencilState = &depthStencil;
		pipelineInfo.pColorBlendState = &colorBlending;// is a pointer to the struct for color blending
		pipelineInfo.layout = pipelineLayout;	// is a description of the binding locations used by the pipeline and the descriptors
		pipelineInfo.renderPass = renderPass;	// is a handle to hold and use the render pass struct previously created
		pipelineInfo.subpass = 0;	// index in which the subpass will be used
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;	// usefull in case we want to derive from a previous pipeline which is not our case

		// Creates the graphics pipeline with the data of the struct and stores it in the global variable for it. Throws a runtime error if it fails
		// The second argument is for a pipeline cache to reuse relevant data for the pipeline in different calls or even program executions if it is stored in a file
		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}

		vkDestroyShaderModule(device, fragShaderModule, nullptr);	// Destroy the fragment shader module
		vkDestroyShaderModule(device, vertShaderModule, nullptr);	// Destroy the vertex shader module
	}

	void createcomputeH0kPipeline() {
		auto computeShaderCode = readFile("computeH0k.spv");	// Read the already compiled vertex shader file

		VkShaderModule computeShaderModule = createShaderModule(computeShaderCode);	// Wrap the bytecode of the vertex shader code into a VkShaderModule object

		VkPipelineShaderStageCreateInfo computeShaderStageInfo = {};
		computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;	// set the type of the struct
		computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;	// specify which stage of the pipeline is defining
		computeShaderStageInfo.module = computeShaderModule;	// Contains the shader object
		computeShaderStageInfo.pName = "main";	// specify the entry point of the shader

		// Here it is specified which uniform variables will be used in the shaders to pass information
		VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;	// set the type of the struct
		pipelineLayoutInfo.setLayoutCount = 1;	// is the number of descriptors for the layout
		pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

		
		VkPushConstantRange pushConstantRange = {};
		pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		pushConstantRange.offset = 0;
		pushConstantRange.size = sizeof(pushConstantsComputeH0kPipeline);

		pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
		pipelineLayoutInfo.pushConstantRangeCount = 1;
		
		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &computeH0kPipelineLayout) != VK_SUCCESS) {	// Creates the layout of the pipeline with the data of the struct
																											// Throws a runtime error if it fails
			throw std::runtime_error("failed to create pipeline layout!");
		}

		VkComputePipelineCreateInfo pipelineInfo = {};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;	// set the type of the struct
		pipelineInfo.stage = computeShaderStageInfo;
		pipelineInfo.layout = computeH0kPipelineLayout;
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;	// usefull in case we want to derive from a previous pipeline which is not our case

		// Creates the graphics pipeline with the data of the struct and stores it in the global variable for it. Throws a runtime error if it fails
		// The second argument is for a pipeline cache to reuse relevant data for the pipeline in different calls or even program executions if it is stored in a file
		if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computeH0kPipeline) != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}

		vkDestroyShaderModule(device, computeShaderModule, nullptr);	// Destroy the fragment shader module
	}

	void createcomputeHktPipeline() {
		auto computeShaderCode = readFile("computeHkt.spv");	// Read the already compiled vertex shader file

		VkShaderModule computeShaderModule = createShaderModule(computeShaderCode);	// Wrap the bytecode of the vertex shader code into a VkShaderModule object

		VkPipelineShaderStageCreateInfo computeShaderStageInfo = {};
		computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;	// set the type of the struct
		computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;	// specify which stage of the pipeline is defining
		computeShaderStageInfo.module = computeShaderModule;	// Contains the shader object
		computeShaderStageInfo.pName = "main";	// specify the entry point of the shader

		// Here it is specified which uniform variables will be used in the shaders to pass information
		VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;	// set the type of the struct
		pipelineLayoutInfo.setLayoutCount = 1;	// is the number of descriptors for the layout
		pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;


		VkPushConstantRange pushConstantRange = {};
		pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		pushConstantRange.offset = 0;
		pushConstantRange.size = sizeof(pushConstantsComputeHktPipeline);

		pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
		pipelineLayoutInfo.pushConstantRangeCount = 1;

		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &computeHktPipelineLayout) != VK_SUCCESS) {	// Creates the layout of the pipeline with the data of the struct
																											// Throws a runtime error if it fails
			throw std::runtime_error("failed to create pipeline layout!");
		}

		VkComputePipelineCreateInfo pipelineInfo = {};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;	// set the type of the struct
		pipelineInfo.stage = computeShaderStageInfo;
		pipelineInfo.layout = computeHktPipelineLayout;
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;	// usefull in case we want to derive from a previous pipeline which is not our case

		// Creates the graphics pipeline with the data of the struct and stores it in the global variable for it. Throws a runtime error if it fails
		// The second argument is for a pipeline cache to reuse relevant data for the pipeline in different calls or even program executions if it is stored in a file
		if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computeHktPipeline) != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}

		vkDestroyShaderModule(device, computeShaderModule, nullptr);	// Destroy the fragment shader module
	}

	void createcomputeFFTAuxPipeline() {
		auto computeShaderCode = readFile("computeFFTAux.spv");	// Read the already compiled vertex shader file

		VkShaderModule computeShaderModule = createShaderModule(computeShaderCode);	// Wrap the bytecode of the vertex shader code into a VkShaderModule object

		VkPipelineShaderStageCreateInfo computeShaderStageInfo = {};
		computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;	// set the type of the struct
		computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;	// specify which stage of the pipeline is defining
		computeShaderStageInfo.module = computeShaderModule;	// Contains the shader object
		computeShaderStageInfo.pName = "main";	// specify the entry point of the shader

		// Here it is specified which uniform variables will be used in the shaders to pass information
		VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;	// set the type of the struct
		pipelineLayoutInfo.setLayoutCount = 1;	// is the number of descriptors for the layout
		pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;


		VkPushConstantRange pushConstantRange = {};
		pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		pushConstantRange.offset = 0;
		pushConstantRange.size = sizeof(pushConstantsComputeFFTAuxPipeline);

		pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
		pipelineLayoutInfo.pushConstantRangeCount = 1;

		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &computeFFTAuxPipelineLayout) != VK_SUCCESS) {	// Creates the layout of the pipeline with the data of the struct
																											// Throws a runtime error if it fails
			throw std::runtime_error("failed to create pipeline layout!");
		}

		VkComputePipelineCreateInfo pipelineInfo = {};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;	// set the type of the struct
		pipelineInfo.stage = computeShaderStageInfo;
		pipelineInfo.layout = computeFFTAuxPipelineLayout;
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;	// usefull in case we want to derive from a previous pipeline which is not our case

		// Creates the graphics pipeline with the data of the struct and stores it in the global variable for it. Throws a runtime error if it fails
		// The second argument is for a pipeline cache to reuse relevant data for the pipeline in different calls or even program executions if it is stored in a file
		if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computeFFTAuxPipeline) != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}

		vkDestroyShaderModule(device, computeShaderModule, nullptr);	// Destroy the fragment shader module
	}

	void createcomputeFFTHorizontalPipeline() {
		auto computeShaderCode = readFile("computeFFTHorizontal.spv");	// Read the already compiled vertex shader file

		VkShaderModule computeShaderModule = createShaderModule(computeShaderCode);	// Wrap the bytecode of the vertex shader code into a VkShaderModule object

		VkPipelineShaderStageCreateInfo computeShaderStageInfo = {};
		computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;	// set the type of the struct
		computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;	// specify which stage of the pipeline is defining
		computeShaderStageInfo.module = computeShaderModule;	// Contains the shader object
		computeShaderStageInfo.pName = "main";	// specify the entry point of the shader

		// Here it is specified which uniform variables will be used in the shaders to pass information
		VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;	// set the type of the struct
		pipelineLayoutInfo.setLayoutCount = 1;	// is the number of descriptors for the layout
		pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;


		VkPushConstantRange pushConstantRange = {};
		pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		pushConstantRange.offset = 0;
		pushConstantRange.size = sizeof(pushConstantsComputeFFTPipeline);

		pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
		pipelineLayoutInfo.pushConstantRangeCount = 1;

		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &computeFFTHorizontalPipelineLayout) != VK_SUCCESS) {	// Creates the layout of the pipeline with the data of the struct
																											// Throws a runtime error if it fails
			throw std::runtime_error("failed to create pipeline layout!");
		}

		VkComputePipelineCreateInfo pipelineInfo = {};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;	// set the type of the struct
		pipelineInfo.stage = computeShaderStageInfo;
		pipelineInfo.layout = computeFFTHorizontalPipelineLayout;
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;	// usefull in case we want to derive from a previous pipeline which is not our case

		// Creates the graphics pipeline with the data of the struct and stores it in the global variable for it. Throws a runtime error if it fails
		// The second argument is for a pipeline cache to reuse relevant data for the pipeline in different calls or even program executions if it is stored in a file
		if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computeFFTHorizontalPipeline) != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}

		vkDestroyShaderModule(device, computeShaderModule, nullptr);	// Destroy the fragment shader module
	}

	void createcomputeFFTVerticalPipeline() {
		auto computeShaderCode = readFile("computeFFTVertical.spv");	// Read the already compiled vertex shader file

		VkShaderModule computeShaderModule = createShaderModule(computeShaderCode);	// Wrap the bytecode of the vertex shader code into a VkShaderModule object

		VkPipelineShaderStageCreateInfo computeShaderStageInfo = {};
		computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;	// set the type of the struct
		computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;	// specify which stage of the pipeline is defining
		computeShaderStageInfo.module = computeShaderModule;	// Contains the shader object
		computeShaderStageInfo.pName = "main";	// specify the entry point of the shader

		// Here it is specified which uniform variables will be used in the shaders to pass information
		VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;	// set the type of the struct
		pipelineLayoutInfo.setLayoutCount = 1;	// is the number of descriptors for the layout
		pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;


		VkPushConstantRange pushConstantRange = {};
		pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		pushConstantRange.offset = 0;
		pushConstantRange.size = sizeof(pushConstantsComputeFFTPipeline);

		pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
		pipelineLayoutInfo.pushConstantRangeCount = 1;

		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &computeFFTVerticalPipelineLayout) != VK_SUCCESS) {	// Creates the layout of the pipeline with the data of the struct
																											// Throws a runtime error if it fails
			throw std::runtime_error("failed to create pipeline layout!");
		}

		VkComputePipelineCreateInfo pipelineInfo = {};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;	// set the type of the struct
		pipelineInfo.stage = computeShaderStageInfo;
		pipelineInfo.layout = computeFFTVerticalPipelineLayout;
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;	// usefull in case we want to derive from a previous pipeline which is not our case

		// Creates the graphics pipeline with the data of the struct and stores it in the global variable for it. Throws a runtime error if it fails
		// The second argument is for a pipeline cache to reuse relevant data for the pipeline in different calls or even program executions if it is stored in a file
		if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computeFFTVerticalPipeline) != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}

		vkDestroyShaderModule(device, computeShaderModule, nullptr);	// Destroy the fragment shader module
	}

	void createcomputeHeightMapPipeline() {
		auto computeShaderCode = readFile("computeHeightMap.spv");	// Read the already compiled vertex shader file

		VkShaderModule computeShaderModule = createShaderModule(computeShaderCode);	// Wrap the bytecode of the vertex shader code into a VkShaderModule object

		VkPipelineShaderStageCreateInfo computeShaderStageInfo = {};
		computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;	// set the type of the struct
		computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;	// specify which stage of the pipeline is defining
		computeShaderStageInfo.module = computeShaderModule;	// Contains the shader object
		computeShaderStageInfo.pName = "main";	// specify the entry point of the shader

		// Here it is specified which uniform variables will be used in the shaders to pass information
		VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;	// set the type of the struct
		pipelineLayoutInfo.setLayoutCount = 1;	// is the number of descriptors for the layout
		pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;


		VkPushConstantRange pushConstantRange = {};
		pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		pushConstantRange.offset = 0;
		pushConstantRange.size = sizeof(pushConstantsComputeHeightMapPipeline);

		pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
		pipelineLayoutInfo.pushConstantRangeCount = 1;

		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &computeHeightMapPipelineLayout) != VK_SUCCESS) {	// Creates the layout of the pipeline with the data of the struct
																											// Throws a runtime error if it fails
			throw std::runtime_error("failed to create pipeline layout!");
		}

		VkComputePipelineCreateInfo pipelineInfo = {};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;	// set the type of the struct
		pipelineInfo.stage = computeShaderStageInfo;
		pipelineInfo.layout = computeHeightMapPipelineLayout;
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;	// usefull in case we want to derive from a previous pipeline which is not our case

		// Creates the graphics pipeline with the data of the struct and stores it in the global variable for it. Throws a runtime error if it fails
		// The second argument is for a pipeline cache to reuse relevant data for the pipeline in different calls or even program executions if it is stored in a file
		if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computeHeightMapPipeline) != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}

		vkDestroyShaderModule(device, computeShaderModule, nullptr);	// Destroy the fragment shader module
	}

	// create the pipeline to create a quad wiht the shadow map to be visible
	// This method will create all the stages of the graphics pipeline, it will load those stages that are programmable and configure those that are configurable
	void createGraphicsPipelineQuad() {
		auto vertShaderCode = readFile("vertQuad.spv");	// Read the already compiled vertex shader file
		auto fragShaderCode = readFile("fragQuad.spv"); // Read the already compiled fragment shader file

		VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);	// Wrap the bytecode of the vertex shader code into a VkShaderModule object
		VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);	// Wrap the bytecode of the fragment shader code into a VkShaderModule object

		VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;	// set the type of the struct
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;	// specify which stage of the pipeline is defining
		vertShaderStageInfo.module = vertShaderModule;	// Contains the shader object
		vertShaderStageInfo.pName = "main";	// specify the entry point of the shader

		VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
		fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;	// set the type of the struct
		fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;	// specify which stage of the pipeline is defining
		fragShaderStageInfo.module = fragShaderModule;	// Contains the shader object
		fragShaderStageInfo.pName = "main";	// specify the entry point of the shader

		VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };	// Holds all the stages of the pipeline that will be implemented

		auto bindingDescription = Vertex::getBindingDescription();
		auto attributeDescriptions = Vertex::getAttributeDescriptions();

		// It specifies the format in which the data will be pass down to the vertex shader 
		VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;	// set the type of the struct
		vertexInputInfo.vertexBindingDescriptionCount = 1;	// is the number of vertex binding descriptors provided 
		vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());	// is the number of vertex attribute descriptions provided
		vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
		vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

		// It specifies which type of primitive will be drawn and if primitives should restart
		VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;	// set the type of the struct
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;	// the primitive type will be triangles specified by 3 vertices without reuse 
		inputAssembly.primitiveRestartEnable = VK_FALSE;	// disable the restart of primitives types

		// defines a transformation from the image to the framebuffer
		VkViewport viewport = {};
		viewport.x = 0.0f;	// Viewport upper left coordinate
		viewport.y = 0.0f;
		viewport.width = (float)swapChainExtent.width;	// Viewport width
		viewport.height = (float)swapChainExtent.height;	// Viewport height
		viewport.minDepth = 0.0f;	// is the depth range of the viewport where the min value can be greater than the max value. They have to be between 0 and 1
		viewport.maxDepth = 1.0f;

		// defines in which regions pixels will actually be stored
		VkRect2D scissor = {};
		scissor.offset = { 0, 0 };	// specifies the rectangle offset
		scissor.extent = swapChainExtent;	// specifies the rectangle extent using the same extent that in the swap chain

		// Struct to combine the viewport with the scissor 
		VkPipelineViewportStateCreateInfo viewportState = {};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;	// set the type of the struct
		viewportState.viewportCount = 1;	// specifies the number of viewports used by the pipeline
		viewportState.pViewports = &viewport;	// pointer to the viewport that will be used
		viewportState.scissorCount = 1;	// specifies the number of scissors used by the pipeline
		viewportState.pScissors = &scissor;	// pointer to the scissor that will be used

		// Turns the geometry shaped by the vertices and transform it into fragments to later be coloured by the fragment shader
		// Also performs depth test, face culling and the scissor test
		VkPipelineRasterizationStateCreateInfo rasterizer = {};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;	// set the type of the struct
		rasterizer.depthClampEnable = VK_FALSE;	// If it's false it will discard the fragments beyond the near and far plane, if it's true it will clamp them to the limits
		rasterizer.rasterizerDiscardEnable = VK_FALSE;	// If it's true it discard all fragments and doesn't give any output
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;	// there are 3 modes, this one fill in the area that the polygon makes.
		rasterizer.lineWidth = 1.0f;	// Describes the thickness of the lines
		rasterizer.cullMode = VK_CULL_MODE_FRONT_BIT;	// determines the type of culling that will be applied 
		rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;	// specifies the vertex order for the faces
		rasterizer.depthBiasEnable = VK_FALSE;	// enables or disable applying a depth bias 
		rasterizer.depthBiasConstantFactor = 0.0f;
		rasterizer.depthBiasClamp = 0.0f;
		rasterizer.depthBiasSlopeFactor = 0.0f; 

		// Is a way to avoid aliasing, it requires to enable a GPU feature 
		VkPipelineMultisampleStateCreateInfo multisampling = {};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;	// set the type of the struct
		multisampling.sampleShadingEnable = VK_FALSE;	// enables or disable sample shading
		multisampling.rasterizationSamples = msaaSamples;
		multisampling.minSampleShading = 1.0f; 
		multisampling.pSampleMask = nullptr; 
		multisampling.alphaToCoverageEnable = VK_FALSE; 
		multisampling.alphaToOneEnable = VK_FALSE;


		VkPipelineDepthStencilStateCreateInfo depthStencil = {};
		depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthStencil.depthTestEnable = VK_TRUE;
		depthStencil.depthWriteEnable = VK_TRUE;
		depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
		depthStencil.depthBoundsTestEnable = VK_FALSE;
		depthStencil.stencilTestEnable = VK_FALSE;

		// proccess the combination of colors from the fragment shader and the color that was previously in the buffer
		// it does it by attached framebuffer
		VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
		// specifies which channels are open to be modified bitwise
		colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_TRUE;	// if it's set to false it will just use the color out of the fragments shader otherwise it will perform color blending
		colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
		colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
		// perform color blending, but it is set globally
		VkPipelineColorBlendStateCreateInfo colorBlending = {};
		colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;	// set the type of the struct
		colorBlending.logicOpEnable = VK_FALSE;	// Controls wether to apply Logical Operations
		colorBlending.logicOp = VK_LOGIC_OP_COPY;	// Selects the logical operation to apply
		colorBlending.attachmentCount = 1;	// Defines the number of VkPipelineColorBlendAttachmentState objects 
		colorBlending.pAttachments = &colorBlendAttachment;	// pointer to the attachments
		colorBlending.blendConstants[0] = 0.0f;	// constant values that are use while blending for R, G, B, A
		colorBlending.blendConstants[1] = 0.0f;
		colorBlending.blendConstants[2] = 0.0f;
		colorBlending.blendConstants[3] = 0.0f;

		// Here it is specified which uniform variables will be used in the shaders to pass information
		VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;	// set the type of the struct
		pipelineLayoutInfo.setLayoutCount = 1;	// is the number of descriptors for the layout
		pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

		VkPushConstantRange pushConstantRange = {};
		pushConstantRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
		pushConstantRange.offset = 0;
		pushConstantRange.size = sizeof(pushConstantsQuadPipeline);

		pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
		pipelineLayoutInfo.pushConstantRangeCount = 1;

		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayoutQuad) != VK_SUCCESS) {	// Creates the layout of the pipeline with the data of the struct
																											// Throws a runtime error if it fails
			throw std::runtime_error("failed to create pipeline layout!");
		}

		VkGraphicsPipelineCreateInfo pipelineInfo = {};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;	// set the type of the struct
		pipelineInfo.stageCount = 2;	// is the number of stages 
		pipelineInfo.pStages = shaderStages;	// is a pointer to the array of the vertex and fragment shader
		pipelineInfo.pVertexInputState = &vertexInputInfo;	// is a pointer to the struct for the vertex input stage
		pipelineInfo.pInputAssemblyState = &inputAssembly;	// is a pointer to the struct for the input assembly
		pipelineInfo.pViewportState = &viewportState;	// is a pointer to the struct for the viewport
		pipelineInfo.pRasterizationState = &rasterizer;	// is a pointer to the struct for the rasterization stage
		pipelineInfo.pMultisampleState = &multisampling;	// is a pointer to the struct for multisampling
		pipelineInfo.pDepthStencilState = &depthStencil;
		pipelineInfo.pColorBlendState = &colorBlending;// is a pointer to the struct for color blending
		pipelineInfo.layout = pipelineLayoutQuad;	// is a description of the binding locations used by the pipeline and the descriptors
		pipelineInfo.renderPass = renderPass;	// is a handle to hold and use the render pass struct previously created
		pipelineInfo.subpass = 0;	// index in which the subpass will be used
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;	// usefull in case we want to derive from a previous pipeline which is not our case

		// Creates the graphics pipeline with the data of the struct and stores it in the global variable for it. Throws a runtime error if it fails
		// The second argument is for a pipeline cache to reuse relevant data for the pipeline in different calls or even program executions if it is stored in a file
		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipelineQuad) != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}

		vkDestroyShaderModule(device, fragShaderModule, nullptr);	// Destroy the fragment shader module
		vkDestroyShaderModule(device, vertShaderModule, nullptr);	// Destroy the vertex shader module
	}

	// create the pipeline to create a quad wiht the shadow map to be visible
	// This method will create all the stages of the graphics pipeline, it will load those stages that are programmable and configure those that are configurable
	void createGraphicsPipelineCube() {
		auto vertShaderCode = readFile("vertCube.spv");	// Read the already compiled vertex shader file
		auto fragShaderCode = readFile("fragCube.spv"); // Read the already compiled fragment shader file

		VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);	// Wrap the bytecode of the vertex shader code into a VkShaderModule object
		VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);	// Wrap the bytecode of the fragment shader code into a VkShaderModule object

		VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;	// set the type of the struct
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;	// specify which stage of the pipeline is defining
		vertShaderStageInfo.module = vertShaderModule;	// Contains the shader object
		vertShaderStageInfo.pName = "main";	// specify the entry point of the shader

		VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
		fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;	// set the type of the struct
		fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;	// specify which stage of the pipeline is defining
		fragShaderStageInfo.module = fragShaderModule;	// Contains the shader object
		fragShaderStageInfo.pName = "main";	// specify the entry point of the shader

		VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };	// Holds all the stages of the pipeline that will be implemented

		auto bindingDescription = VertexOnlyPos::getBindingDescription();
		auto attributeDescriptions = VertexOnlyPos::getAttributeDescriptions();

		// It specifies the format in which the data will be pass down to the vertex shader 
		VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;	// set the type of the struct
		vertexInputInfo.vertexBindingDescriptionCount = 1;	// is the number of vertex binding descriptors provided 
		vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());	// is the number of vertex attribute descriptions provided
		vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
		vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

		// It specifies which type of primitive will be drawn and if primitives should restart
		VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;	// set the type of the struct
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;	// the primitive type will be triangles specified by 3 vertices without reuse 
		inputAssembly.primitiveRestartEnable = VK_FALSE;	// disable the restart of primitives types

		// defines a transformation from the image to the framebuffer
		VkViewport viewport = {};
		viewport.x = 0.0f;	// Viewport upper left coordinate
		viewport.y = 0.0f;
		viewport.width = (float)swapChainExtent.width;	// Viewport width
		viewport.height = (float)swapChainExtent.height;	// Viewport height
		viewport.minDepth = 0.0f;	// is the depth range of the viewport where the min value can be greater than the max value. They have to be between 0 and 1
		viewport.maxDepth = 1.0f;

		// defines in which regions pixels will actually be stored
		VkRect2D scissor = {};
		scissor.offset = { 0, 0 };	// specifies the rectangle offset
		scissor.extent = swapChainExtent;	// specifies the rectangle extent using the same extent that in the swap chain

		// Struct to combine the viewport with the scissor 
		VkPipelineViewportStateCreateInfo viewportState = {};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;	// set the type of the struct
		viewportState.viewportCount = 1;	// specifies the number of viewports used by the pipeline
		viewportState.pViewports = &viewport;	// pointer to the viewport that will be used
		viewportState.scissorCount = 1;	// specifies the number of scissors used by the pipeline
		viewportState.pScissors = &scissor;	// pointer to the scissor that will be used

		// Turns the geometry shaped by the vertices and transform it into fragments to later be coloured by the fragment shader
		// Also performs depth test, face culling and the scissor test
		VkPipelineRasterizationStateCreateInfo rasterizer = {};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;	// set the type of the struct
		rasterizer.depthClampEnable = VK_FALSE;	// If it's false it will discard the fragments beyond the near and far plane, if it's true it will clamp them to the limits
		rasterizer.rasterizerDiscardEnable = VK_FALSE;	// If it's true it discard all fragments and doesn't give any output
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;	// there are 3 modes, this one fill in the area that the polygon makes.
		rasterizer.lineWidth = 1.0f;	// Describes the thickness of the lines
		rasterizer.cullMode = VK_CULL_MODE_FRONT_BIT;	// determines the type of culling that will be applied 
		rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;	// specifies the vertex order for the faces
		rasterizer.depthBiasEnable = VK_FALSE;	// enables or disable applying a depth bias 
		rasterizer.depthBiasConstantFactor = 0.0f;
		rasterizer.depthBiasClamp = 0.0f;
		rasterizer.depthBiasSlopeFactor = 0.0f;

		// Is a way to avoid aliasing, it requires to enable a GPU feature 
		VkPipelineMultisampleStateCreateInfo multisampling = {};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;	// set the type of the struct
		multisampling.sampleShadingEnable = VK_FALSE;	// enables or disable sample shading
		multisampling.rasterizationSamples = msaaSamples;
		multisampling.minSampleShading = 1.0f;
		multisampling.pSampleMask = nullptr;
		multisampling.alphaToCoverageEnable = VK_FALSE;
		multisampling.alphaToOneEnable = VK_FALSE;


		VkPipelineDepthStencilStateCreateInfo depthStencil = {};
		depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthStencil.depthTestEnable = VK_TRUE;
		depthStencil.depthWriteEnable = VK_TRUE;
		depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
		depthStencil.depthBoundsTestEnable = VK_FALSE;
		depthStencil.stencilTestEnable = VK_FALSE;

		// proccess the combination of colors from the fragment shader and the color that was previously in the buffer
		// it does it by attached framebuffer
		VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
		// specifies which channels are open to be modified bitwise
		colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_FALSE;	// if it's set to false it will just use the color out of the fragments shader otherwise it will perform color blending
		colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
		colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
		// perform color blending, but it is set globally
		VkPipelineColorBlendStateCreateInfo colorBlending = {};
		colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;	// set the type of the struct
		colorBlending.logicOpEnable = VK_FALSE;	// Controls wether to apply Logical Operations
		colorBlending.logicOp = VK_LOGIC_OP_COPY;	// Selects the logical operation to apply
		colorBlending.attachmentCount = 1;	// Defines the number of VkPipelineColorBlendAttachmentState objects 
		colorBlending.pAttachments = &colorBlendAttachment;	// pointer to the attachments
		colorBlending.blendConstants[0] = 0.0f;	// constant values that are use while blending for R, G, B, A
		colorBlending.blendConstants[1] = 0.0f;
		colorBlending.blendConstants[2] = 0.0f;
		colorBlending.blendConstants[3] = 0.0f;

		// Here it is specified which uniform variables will be used in the shaders to pass information
		VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;	// set the type of the struct
		pipelineLayoutInfo.setLayoutCount = 1;	// is the number of descriptors for the layout
		pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayoutCube) != VK_SUCCESS) {	// Creates the layout of the pipeline with the data of the struct
																											// Throws a runtime error if it fails
			throw std::runtime_error("failed to create pipeline layout!");
		}

		VkGraphicsPipelineCreateInfo pipelineInfo = {};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;	// set the type of the struct
		pipelineInfo.stageCount = 2;	// is the number of stages 
		pipelineInfo.pStages = shaderStages;	// is a pointer to the array of the vertex and fragment shader
		pipelineInfo.pVertexInputState = &vertexInputInfo;	// is a pointer to the struct for the vertex input stage
		pipelineInfo.pInputAssemblyState = &inputAssembly;	// is a pointer to the struct for the input assembly
		pipelineInfo.pViewportState = &viewportState;	// is a pointer to the struct for the viewport
		pipelineInfo.pRasterizationState = &rasterizer;	// is a pointer to the struct for the rasterization stage
		pipelineInfo.pMultisampleState = &multisampling;	// is a pointer to the struct for multisampling
		pipelineInfo.pDepthStencilState = &depthStencil;
		pipelineInfo.pColorBlendState = &colorBlending;// is a pointer to the struct for color blending
		pipelineInfo.layout = pipelineLayoutCube;	// is a description of the binding locations used by the pipeline and the descriptors
		pipelineInfo.renderPass = renderPass;	// is a handle to hold and use the render pass struct previously created
		pipelineInfo.subpass = 0;	// index in which the subpass will be used
		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;	// usefull in case we want to derive from a previous pipeline which is not our case

		// Creates the graphics pipeline with the data of the struct and stores it in the global variable for it. Throws a runtime error if it fails
		// The second argument is for a pipeline cache to reuse relevant data for the pipeline in different calls or even program executions if it is stored in a file
		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipelineCube) != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline!");
		}

		vkDestroyShaderModule(device, fragShaderModule, nullptr);	// Destroy the fragment shader module
		vkDestroyShaderModule(device, vertShaderModule, nullptr);	// Destroy the vertex shader module
	}

	// Create a new frame buffer for the gui
	// Create the frame buffer that will hold all of the images of the swap chain and will wrap up all of the others buffers
	void createFramebuffersGUI() {
		swapChainFramebuffersGUI.resize(swapChainImageViews.size());	// Resize the buffer to be able to contain all the images of the swap chain

		for (size_t i = 0; i < swapChainImageViews.size(); i++) {	// iterate through every image of the swap chain and create a buffer for each
			std::array<VkImageView, 1> attachments = {
				swapChainImageViews[i]
			};

			VkFramebufferCreateInfo framebufferInfo = {};
			framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;	// set the type of the struct
			framebufferInfo.renderPass = renderPassGUI;	// indicates which renderPass will be used and how it is configured 
			framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());	// We will only use one attachments to the buffer that is the color buffer
			framebufferInfo.pAttachments = attachments.data();	// pointer to the attachments to use in the buffer
			framebufferInfo.width = swapChainExtent.width;	// indicates the size of the images in the swap chain
			framebufferInfo.height = swapChainExtent.height;
			framebufferInfo.layers = 1;	// refers to the number of layers in every image of the swap chain

			if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffersGUI[i]) != VK_SUCCESS) {	// Create the FrameBuffer object using the data of the struct
																													// Throw a runtime error exception if something fails
				throw std::runtime_error("failed to create framebuffer!");
			}
		}
	}


	// Create the frame buffer that will hold all of the images of the swap chain and will wrap up all of the others buffers
	void createFramebuffers() {
		swapChainFramebuffers.resize(swapChainImageViews.size());	// Resize the buffer to be able to contain all the images of the swap chain

		for (size_t i = 0; i < swapChainImageViews.size(); i++) {	// iterate through every image of the swap chain and create a buffer for each
			
			VkFramebufferCreateInfo framebufferInfo = {};

			if (msaaSamples != VK_SAMPLE_COUNT_1_BIT)
			{
				std::array<VkImageView, 3> attachments = {
				maaImageView,
				depthImageView,
				swapChainImageViews[i]
				};


				framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;	// set the type of the struct
				framebufferInfo.renderPass = renderPass;	// indicates which renderPass will be used and how it is configured 
				framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());	// We will only use one attachments to the buffer that is the color buffer
				framebufferInfo.pAttachments = attachments.data();	// pointer to the attachments to use in the buffer
				framebufferInfo.width = swapChainExtent.width;	// indicates the size of the images in the swap chain
				framebufferInfo.height = swapChainExtent.height;
				framebufferInfo.layers = 1;	// refers to the number of layers in every image of the swap chain
			}
			else
			{
				std::array<VkImageView, 2> attachments = {
				swapChainImageViews[i],
				depthImageView
				};


				framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;	// set the type of the struct
				framebufferInfo.renderPass = renderPass;	// indicates which renderPass will be used and how it is configured 
				framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());	// We will only use one attachments to the buffer that is the color buffer
				framebufferInfo.pAttachments = attachments.data();	// pointer to the attachments to use in the buffer
				framebufferInfo.width = swapChainExtent.width;	// indicates the size of the images in the swap chain
				framebufferInfo.height = swapChainExtent.height;
				framebufferInfo.layers = 1;	// refers to the number of layers in every image of the swap chain
			}
			

			if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {	// Create the FrameBuffer object using the data of the struct
																													// Throw a runtime error exception if something fails
				throw std::runtime_error("failed to create framebuffer!");
			}
		}
	}


	// Create a manager of the memory that will store the buffers and command buffers
	void createCommandPool() {
		QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);	// first we need to find the index of the different queues available

		VkCommandPoolCreateInfo poolInfo = {};
		poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;	// set the type of the struct
		poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();	// as we want to use the draw call, we will need the graphics queue

		if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {	// Create the command pool using the data of the struct 
																							// Throw a runtime error if something fails
			throw std::runtime_error("failed to create command pool!");
		}
	}

	void createMaaResources() {
		VkFormat colorFormat = swapChainImageFormat;

		createImage(swapChainExtent.width, swapChainExtent.height, colorFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, maaImage, maaImageMemory, msaaSamples);
		maaImageView = createImageView(maaImage, colorFormat, VK_IMAGE_ASPECT_COLOR_BIT);
	}

	// Create all the info required for the depth buffer to work
	void createDepthResources() {
		VkFormat depthFormat = findDepthFormat();	// search for an adequate format

		createImage(swapChainExtent.width, swapChainExtent.height, depthFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage, depthImageMemory, msaaSamples);
		depthImageView = createImageView(depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);	// Create an image view that will be used for the depth
	}

	void createComputeShaderResources() {

		createImage(fourierGridSize, fourierGridSize, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, computeShaderImageHk0, computeShaderImageMemoryHk0);
		transitionImageLayout(computeShaderImageHk0, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,1);
		computeShaderImageViewHk0 = createImageView(computeShaderImageHk0, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT);
		
		createImage(fourierGridSize, fourierGridSize, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, computeShaderImageHk0minus, computeShaderImageMemoryHk0minus);
		transitionImageLayout(computeShaderImageHk0minus, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,1);
		computeShaderImageViewHk0minus = createImageView(computeShaderImageHk0minus, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT);

		createImage(fourierGridSize, fourierGridSize, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, computeShaderImageHkt, computeShaderImageMemoryHkt);
		transitionImageLayout(computeShaderImageHkt, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,1);
		computeShaderImageViewHkt = createImageView(computeShaderImageHkt, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT);

		createImage(int(log2(fourierGridSize)), fourierGridSize, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, computeShaderImageFFTAux, computeShaderImageMemoryFFTAux);
		transitionImageLayout(computeShaderImageFFTAux, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,1);
		computeShaderImageViewFFTAux = createImageView(computeShaderImageFFTAux, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT);

		createImage(fourierGridSize, fourierGridSize, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, computeShaderImageFFTAlternate, computeShaderImageMemoryFFTAlternate);
		transitionImageLayout(computeShaderImageFFTAlternate, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,1);
		computeShaderImageViewFFTAlternate = createImageView(computeShaderImageFFTAlternate, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT);

		createImage(fourierGridSize, fourierGridSize, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, computeShaderImageHeightMap, computeShaderImageMemoryHeightMap);
		transitionImageLayout(computeShaderImageHeightMap, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,1);
		computeShaderImageViewHeightMap = createImageView(computeShaderImageHeightMap, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT);

		createImage(fourierGridSize, fourierGridSize, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, computeShaderImageSlopeX, computeShaderImageMemorySlopeX);
		transitionImageLayout(computeShaderImageSlopeX, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,1);
		computeShaderImageViewSlopeX = createImageView(computeShaderImageSlopeX, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT);

		createImage(fourierGridSize, fourierGridSize, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, computeShaderImageSlopeXAlternate, computeShaderImageMemorySlopeXAlternate);
		transitionImageLayout(computeShaderImageSlopeXAlternate, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,1);
		computeShaderImageViewSlopeXAlternate = createImageView(computeShaderImageSlopeXAlternate, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT);

		createImage(fourierGridSize, fourierGridSize, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, computeShaderImageSlopeXFinal, computeShaderImageMemorySlopeXFinal);
		transitionImageLayout(computeShaderImageSlopeXFinal, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,1);
		computeShaderImageViewSlopeXFinal = createImageView(computeShaderImageSlopeXFinal, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT);

		createImage(fourierGridSize, fourierGridSize, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, computeShaderImageSlopeZ, computeShaderImageMemorySlopeZ);
		transitionImageLayout(computeShaderImageSlopeZ, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,1);
		computeShaderImageViewSlopeZ = createImageView(computeShaderImageSlopeZ, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT);

		createImage(fourierGridSize, fourierGridSize, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, computeShaderImageSlopeZAlternate, computeShaderImageMemorySlopeZAlternate);
		transitionImageLayout(computeShaderImageSlopeZAlternate, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,1);
		computeShaderImageViewSlopeZAlternate = createImageView(computeShaderImageSlopeZAlternate, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT);

		createImage(fourierGridSize, fourierGridSize, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, computeShaderImageSlopeZFinal, computeShaderImageMemorySlopeZFinal);
		transitionImageLayout(computeShaderImageSlopeZFinal, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,1);
		computeShaderImageViewSlopeZFinal = createImageView(computeShaderImageSlopeZFinal, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT);

		createImage(fourierGridSize, fourierGridSize, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, computeShaderImageDispX, computeShaderImageMemoryDispX);
		transitionImageLayout(computeShaderImageDispX, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,1);
		computeShaderImageViewDispX = createImageView(computeShaderImageDispX, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT);

		createImage(fourierGridSize, fourierGridSize, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, computeShaderImageDispXAlternate, computeShaderImageMemoryDispXAlternate);
		transitionImageLayout(computeShaderImageDispXAlternate, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,1);
		computeShaderImageViewDispXAlternate = createImageView(computeShaderImageDispXAlternate, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT);

		createImage(fourierGridSize, fourierGridSize, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, computeShaderImageDispXFinal, computeShaderImageMemoryDispXFinal);
		transitionImageLayout(computeShaderImageDispXFinal, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,1);
		computeShaderImageViewDispXFinal = createImageView(computeShaderImageDispXFinal, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT);

		createImage(fourierGridSize, fourierGridSize, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, computeShaderImageDispZ, computeShaderImageMemoryDispZ);
		transitionImageLayout(computeShaderImageDispZ, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,1);
		computeShaderImageViewDispZ = createImageView(computeShaderImageDispZ, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT);

		createImage(fourierGridSize, fourierGridSize, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, computeShaderImageDispZAlternate, computeShaderImageMemoryDispZAlternate);
		transitionImageLayout(computeShaderImageDispZAlternate, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,1);
		computeShaderImageViewDispZAlternate = createImageView(computeShaderImageDispZAlternate, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT);

		createImage(fourierGridSize, fourierGridSize, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, computeShaderImageDispZFinal, computeShaderImageMemoryDispZFinal);
		transitionImageLayout(computeShaderImageDispZFinal, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,1);
		computeShaderImageViewDispZFinal = createImageView(computeShaderImageDispZFinal, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT);
	}


	// find the adequate format for the depth buffer to work
	VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {
		for (VkFormat format : candidates) {
			VkFormatProperties props;
			vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);

			if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
				return format;
			}
			else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
				return format;
			}
		}

		throw std::runtime_error("failed to find supported format!");
	}
	// Calls a method with the hardcoded parameters to find a suitable format
	VkFormat findDepthFormat() {
		return findSupportedFormat(
			{ VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
			VK_IMAGE_TILING_OPTIMAL,
			VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
		);
	}

	bool hasStencilComponent(VkFormat format) {
		return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
	}

	void loadTextureCubeMap(std::string fileName, unsigned char * pixels, int numberTexture)
	{
		int texWidth, texHeight;

		std::ifstream inFile;
		inFile.open(fileName, std::ios::binary);
		if (!inFile) {
			throw std::runtime_error("Error opening Model object");
		}

		std::string line;
		int aux = 0;
		std::getline(inFile, line);
		std::getline(inFile, line);
		inFile >> texWidth;
		inFile >> texHeight;
		inFile >> aux;

		int offset = texWidth * texHeight * 4 * numberTexture;

		inFile.get();
		char x, y, z = (unsigned char)255;
		for (int i = 0; i < texWidth*texHeight; i++) {
			inFile.read(&x, sizeof(char));
			pixels[(i * 4) + offset] = x;

			inFile.read(&y, sizeof(char));
			pixels[(i * 4) + 1 + offset] = y;

			inFile.read(&z, sizeof(char));
			pixels[(i * 4) + 2 + offset] = z;

			pixels[(i * 4) + 3 + offset] = (unsigned char)255;

		}
		inFile.close();

	}

	VkSampleCountFlagBits getMaxUsableSampleCount() {
		VkPhysicalDeviceProperties physicalDeviceProperties;
		vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);

		VkSampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts & physicalDeviceProperties.limits.framebufferDepthSampleCounts;
		if (counts & VK_SAMPLE_COUNT_64_BIT) { return VK_SAMPLE_COUNT_64_BIT; }
		if (counts & VK_SAMPLE_COUNT_32_BIT) { return VK_SAMPLE_COUNT_32_BIT; }
		if (counts & VK_SAMPLE_COUNT_16_BIT) { return VK_SAMPLE_COUNT_16_BIT; }
		if (counts & VK_SAMPLE_COUNT_8_BIT) { return VK_SAMPLE_COUNT_8_BIT; }
		if (counts & VK_SAMPLE_COUNT_4_BIT) { return VK_SAMPLE_COUNT_4_BIT; }
		if (counts & VK_SAMPLE_COUNT_2_BIT) { return VK_SAMPLE_COUNT_2_BIT; }

		return VK_SAMPLE_COUNT_1_BIT;
	}


	// Load the texture.
	void createTextureImage() {
		int texWidth, texHeight;

		std::ifstream inFile;
		inFile.open("Textures/Camping/posx.ppm", std::ios::binary);	// If we change all the folder name Camping to skybox we can see a different cubemap image
		if (!inFile) {
			throw std::runtime_error("Error opening Model object");
		}

		std::string line;
		int aux = 0;
		std::getline(inFile, line);
		std::getline(inFile, line);
		inFile >> texWidth;
		inFile >> texHeight;
		inFile >> aux;
		VkDeviceSize imageSize = texWidth * texHeight * 4 * 6;

		unsigned char * pixels = new unsigned char[(size_t)imageSize];

		inFile.get();
		char x,y,z = (unsigned char)255;
		for (int i = 0; i < texWidth*texHeight; i++) {
			inFile.read(&x, sizeof(char));
			pixels[(i * 4)] = x;

			inFile.read(&y, sizeof(char));
			pixels[(i * 4)+1] = y;

			inFile.read(&z, sizeof(char));
			pixels[(i * 4)+2] = z;

			pixels[(i * 4)+3] = (unsigned char) 255;

		}
		inFile.close();
		loadTextureCubeMap("Textures/Camping/negx.ppm", pixels, 1);
		loadTextureCubeMap("Textures/Camping/posy.ppm", pixels, 2);
		loadTextureCubeMap("Textures/Camping/negy.ppm", pixels, 3);
		loadTextureCubeMap("Textures/Camping/posz.ppm", pixels, 4);
		loadTextureCubeMap("Textures/Camping/negz.ppm", pixels, 5);

		// Store the data in an staging buffer
		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
 		createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		// Map the data 
		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
		memcpy(data, pixels, static_cast<size_t>(imageSize));
		vkUnmapMemory(device, stagingBufferMemory);

		delete[] pixels;

		VkImageCreateInfo imageInfo = {};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.extent.width = texWidth;
		imageInfo.extent.height = texHeight;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 6;
		imageInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		imageInfo.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;

		if (vkCreateImage(device, &imageInfo, nullptr, &textureImage) != VK_SUCCESS) {
			throw std::runtime_error("failed to create image!");
		}

		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(device, textureImage, &memRequirements);

		VkMemoryAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

		if (vkAllocateMemory(device, &allocInfo, nullptr, &textureImageMemory) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate image memory!");
		}

		vkBindImageMemory(device, textureImage, textureImageMemory, 0);

		// execute commands buffers
		transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,6);
		
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkBufferImageCopy region = {};
		region.bufferOffset = 0;
		region.bufferRowLength = 0;
		region.bufferImageHeight = 0;
		region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		region.imageSubresource.mipLevel = 0;
		region.imageSubresource.baseArrayLayer = 0;
		region.imageSubresource.layerCount = 6;
		region.imageOffset = { 0, 0, 0 };
		region.imageExtent = {
			static_cast<uint32_t>(texWidth),
			static_cast<uint32_t>(texHeight),
			1
		};

		vkCmdCopyBufferToImage(commandBuffer, stagingBuffer, textureImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

		endSingleTimeCommands(commandBuffer);
		

		
		//copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
		transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,6);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}
	void createTextureImageView() {

		VkImageViewCreateInfo viewInfo = {};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = textureImage;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
		viewInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
		viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		viewInfo.subresourceRange.baseMipLevel = 0;
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount = 6;

		if (vkCreateImageView(device, &viewInfo, nullptr, &textureImageView) != VK_SUCCESS) {
			throw std::runtime_error("failed to create texture image view!");
		}
	}

	// Creates a sampler for the texture
	void createTextureSamplerCube() {
		VkSamplerCreateInfo samplerInfo = {};
		samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerInfo.magFilter = VK_FILTER_LINEAR;
		samplerInfo.minFilter = VK_FILTER_LINEAR;
		samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;	// VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE
		samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;	// VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE
		samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;	// VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE
		samplerInfo.anisotropyEnable = VK_TRUE;
		samplerInfo.maxAnisotropy = 16;	// 1.0f
		samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
		samplerInfo.unnormalizedCoordinates = VK_FALSE;
		samplerInfo.compareEnable = VK_FALSE;
		samplerInfo.compareOp = VK_COMPARE_OP_NEVER;
		samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		samplerInfo.mipLodBias = 0.0f;
		samplerInfo.minLod = 0.0f;
		samplerInfo.maxLod = 0.0f;

		if (vkCreateSampler(device, &samplerInfo, nullptr, &textureSamplerCube) != VK_SUCCESS) {
			throw std::runtime_error("failed to create texture sampler!");
		}
	}

	// Creates a sampler for the texture
	void createTextureSampler() {
		VkSamplerCreateInfo samplerInfo = {};
		samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerInfo.magFilter = VK_FILTER_LINEAR;
		samplerInfo.minFilter = VK_FILTER_LINEAR;
		samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;	// VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE
		samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;	// VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE
		samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;	// VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE
		samplerInfo.anisotropyEnable = VK_TRUE;
		samplerInfo.maxAnisotropy = 16;	// 1.0f
		samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
		samplerInfo.unnormalizedCoordinates = VK_FALSE;
		samplerInfo.compareEnable = VK_FALSE;
		samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
		samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		samplerInfo.mipLodBias = 0.0f;
		samplerInfo.minLod = 0.0f;
		samplerInfo.maxLod = 0.0f;

		if (vkCreateSampler(device, &samplerInfo, nullptr, &textureSampler) != VK_SUCCESS) {
			throw std::runtime_error("failed to create texture sampler!");
		}
	}

	// Create an image view depending on the type of flag
	VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags) {
		VkImageViewCreateInfo viewInfo = {};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = format;
		viewInfo.subresourceRange.aspectMask = aspectFlags;
		viewInfo.subresourceRange.baseMipLevel = 0;
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount = 1;

		VkImageView imageView;
		if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
			throw std::runtime_error("failed to create texture image view!");
		}

		return imageView;
	}

	// Creates an image with the data of the parameters. Also check for the memory requirements and allocate space for that image
	void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageLayout layout ,VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory) {
		VkImageCreateInfo imageInfo = {};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.extent.width = width;
		imageInfo.extent.height = height;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.format = format; 
		imageInfo.tiling = tiling;
		imageInfo.initialLayout = layout;
		imageInfo.usage = usage;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
			throw std::runtime_error("failed to create image!");
		}

		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(device, image, &memRequirements);

		VkMemoryAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

		if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate image memory!");
		}

		vkBindImageMemory(device, image, imageMemory, 0);
	}

	// Creates an image with the data of the parameters. Also check for the memory requirements and allocate space for that image
	void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageLayout layout, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory, VkSampleCountFlagBits numSample) {
		VkImageCreateInfo imageInfo = {};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.extent.width = width;
		imageInfo.extent.height = height;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.format = format;
		imageInfo.tiling = tiling;
		imageInfo.initialLayout = layout;
		imageInfo.usage = usage;
		imageInfo.samples = numSample;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
			throw std::runtime_error("failed to create image!");
		}

		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(device, image, &memRequirements);

		VkMemoryAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

		if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate image memory!");
		}

		vkBindImageMemory(device, image, imageMemory, 0);
	}


	// Execute commands buffers
	void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t layerCount) {
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkImageMemoryBarrier barrier = {};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.oldLayout = oldLayout;
		barrier.newLayout = newLayout;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.image = image;
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		barrier.subresourceRange.baseMipLevel = 0;
		barrier.subresourceRange.levelCount = 1;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = layerCount;

		VkPipelineStageFlags sourceStage;
		VkPipelineStageFlags destinationStage;

		if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
			barrier.srcAccessMask = 0; //no src access mask as it is undefined
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; //dst access mask is a write, because we are copying it to

			sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT; //the stage before the barrier is top of pipeline
			destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT; //after the barrier it is transfer stage
		}
		//if we are taking the image from the transfer layout to the shader read only optimal layout
		else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; //before barrier we have write access
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT; //after the barrier we have shader only access 

			sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT; //before the barrier we have the transfer stage
			destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT; //after the barrier we have the fragment shader stage
		}
		else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_GENERAL) {
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

			sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			destinationStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
		}
		else {
			throw std::invalid_argument("unsupported layout transition!");
		}

		vkCmdPipelineBarrier(
			commandBuffer,
			sourceStage, destinationStage,
			0,
			0, nullptr,
			0, nullptr,
			1, &barrier
		);

		endSingleTimeCommands(commandBuffer);
	}

	// Perform a copy of the information in the buffer to the image
	void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkBufferImageCopy region = {};
		region.bufferOffset = 0;
		region.bufferRowLength = 0;
		region.bufferImageHeight = 0;
		region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		region.imageSubresource.mipLevel = 0;
		region.imageSubresource.baseArrayLayer = 0;
		region.imageSubresource.layerCount = 1;
		region.imageOffset = { 0, 0, 0 };
		region.imageExtent = {
			width,
			height,
			1
		};

		vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

		endSingleTimeCommands(commandBuffer);
	}

	// fill in the array of vertices and the array of indices using an Obj file
	void loadModel() {		
		
		vertices.clear();
		indices.clear();
		indicesQuad.clear();
		verticesQuad.clear();
		verticesCube.clear();
		indicesCube.clear();

		//CENTER

		for (int i = 0; i < fourierGridSize; i++)
		{
			for (int j = 0; j < fourierGridSize; j++)
			{
				Vertex v = {};		

				v.color = waterColor;
				v.pos = glm::vec3(i, 0.0f, j);
				v.texCoord = glm::vec2(i / float(fourierGridSize - 1), j / float(fourierGridSize - 1));
				v.normalCoord = glm::vec3(0.0f, 1.0f, 0.0f);

				if (i - 1 >= 0 && j - 1 >= 0)
				{
					indices.push_back(i * fourierGridSize + j);
					indices.push_back(i * fourierGridSize + (j - 1));
					indices.push_back((i - 1) * fourierGridSize + j);
				}

				if (i + 1 <= fourierGridSize - 1 && j + 1 <= fourierGridSize - 1)
				{
					indices.push_back(i * fourierGridSize + j);
					indices.push_back(i * fourierGridSize + (j + 1));
					indices.push_back((i + 1) * fourierGridSize + j);
				}
				vertices.push_back(v);
			}
		}
		
		

		int fourierGridSizeX2 = fourierGridSize * fourierGridSize;
		int timesGrid = 1;
		
		// LEFT

		for (int i = 0; i < fourierGridSize; i++)
		{
			for (int j = 0; j < fourierGridSize; j++)
			{
				Vertex v = {};

				v.color = waterColor;
				v.pos = glm::vec3(i + fourierGridSize - 1, 0.0f, j);

				if (i == 0 && j == 0)
				{
					v.texCoord = glm::vec2(1,0);
				}
				else if (i == 0 && j == fourierGridSize - 1)
				{
					v.texCoord = glm::vec2(1, 1);
				}
				else if (i == 0)
				{
					v.texCoord = glm::vec2(1 - (i / float(fourierGridSize - 1)), j / float(fourierGridSize - 1));
				}
				else if (j == fourierGridSize - 1)
				{
					v.texCoord = glm::vec2(i / float(fourierGridSize - 1), 1 - (j / float(fourierGridSize - 1)));
				}
				else
				{
					v.texCoord = glm::vec2(i / float(fourierGridSize - 1), j / float(fourierGridSize - 1));
				}
				v.normalCoord = glm::vec3(0.0f, 1.0f, 0.0f);

				if (i - 1 >= 0 && j - 1 >= 0)
				{
					indices.push_back(i * fourierGridSize + j + fourierGridSizeX2 * timesGrid);
					indices.push_back(i * fourierGridSize + (j - 1) + fourierGridSizeX2 * timesGrid);
					indices.push_back((i - 1) * fourierGridSize + j + fourierGridSizeX2 * timesGrid);
				}

				if (i + 1 <= fourierGridSize - 1 && j + 1 <= fourierGridSize - 1)
				{
					indices.push_back(i * fourierGridSize + j + fourierGridSizeX2 * timesGrid);
					indices.push_back(i * fourierGridSize + (j + 1) + fourierGridSizeX2 * timesGrid);
					indices.push_back((i + 1) * fourierGridSize + j + fourierGridSizeX2 * timesGrid);
				}
				vertices.push_back(v);
			}
		}
		timesGrid++;
		

		// UP LEFT

		for (int i = 0; i < fourierGridSize; i++)
		{
			for (int j = 0; j < fourierGridSize; j++)
			{
				Vertex v = {};

				v.color = waterColor;
				v.pos = glm::vec3(i + fourierGridSize-1, 0.0f, j + fourierGridSize-1);
				if (i == 0 && j == 0)
				{
					v.texCoord = glm::vec2(1,1);
				}
				else if (i == 0)
				{
					v.texCoord = glm::vec2(1 - (i / float(fourierGridSize - 1)), j / float(fourierGridSize - 1));
				}
				else
				{
					v.texCoord = glm::vec2(i / float(fourierGridSize - 1), j / float(fourierGridSize - 1));
				}
				v.normalCoord = glm::vec3(0.0f, 1.0f, 0.0f);

				if (i - 1 >= 0 && j - 1 >= 0)
				{
					indices.push_back(i * fourierGridSize + j + fourierGridSizeX2 * timesGrid);
					indices.push_back(i * fourierGridSize + (j - 1) + fourierGridSizeX2 * timesGrid);
					indices.push_back((i - 1) * fourierGridSize + j + fourierGridSizeX2 * timesGrid);
				}

				if (i + 1 <= fourierGridSize - 1 && j + 1 <= fourierGridSize - 1)
				{
					indices.push_back(i * fourierGridSize + j + fourierGridSizeX2 * timesGrid);
					indices.push_back(i * fourierGridSize + (j + 1) + fourierGridSizeX2 * timesGrid);
					indices.push_back((i + 1) * fourierGridSize + j + fourierGridSizeX2 * timesGrid);
				}
				vertices.push_back(v);
			}
		}
		timesGrid++;

		
		
		// UP

		for (int i = 0; i < fourierGridSize; i++)
		{
			for (int j = 0; j < fourierGridSize; j++)
			{
				Vertex v = {};

				v.color = waterColor;
				v.pos = glm::vec3(i, 0.0f, j + fourierGridSize -1);
				
				if (i == 0 && j == 0)
				{
					v.texCoord = glm::vec2(0, 1);
				}
				else if (i == fourierGridSize -1 && j == 0)
				{
					v.texCoord = glm::vec2(1, 1);
				}
				else if (i == 0)
				{
					v.texCoord = glm::vec2(1 - (i / float(fourierGridSize - 1)), j / float(fourierGridSize - 1));
				}
				else if (j == 0)
				{
					v.texCoord = glm::vec2(i / float(fourierGridSize - 1), 1 - (j / float(fourierGridSize - 1)));
				}
				else
				{
					v.texCoord = glm::vec2(i / float(fourierGridSize - 1), j / float(fourierGridSize - 1));
				}
				v.normalCoord = glm::vec3(0.0f, 1.0f, 0.0f);

				if (i - 1 >= 0 && j - 1 >= 0)
				{
					indices.push_back(i * fourierGridSize + j + fourierGridSizeX2 * timesGrid);
					indices.push_back(i * fourierGridSize + (j - 1) + fourierGridSizeX2 * timesGrid);
					indices.push_back((i - 1) * fourierGridSize + j + fourierGridSizeX2 * timesGrid);
				}

				if (i + 1 <= fourierGridSize - 1 && j + 1 <= fourierGridSize - 1)
				{
					indices.push_back(i * fourierGridSize + j + fourierGridSizeX2 * timesGrid);
					indices.push_back(i * fourierGridSize + (j + 1) + fourierGridSizeX2 * timesGrid);
					indices.push_back((i + 1) * fourierGridSize + j + fourierGridSizeX2 * timesGrid);
				}
				vertices.push_back(v);
			}
		}
		timesGrid++;

		// RIGHT UP

		for (int i = 0; i < fourierGridSize; i++)
		{
			for (int j = 0; j < fourierGridSize; j++)
			{
				Vertex v = {};

				v.color = waterColor;
				v.pos = glm::vec3(i - fourierGridSize + 1, 0.0f, j + fourierGridSize - 1);

				if (i == fourierGridSize - 1 && j == 0)
				{
					v.texCoord = glm::vec2(0,1);
				}
				else if (j == 0)
				{
					v.texCoord = glm::vec2(i / float(fourierGridSize - 1), 1 - (j / float(fourierGridSize - 1)));
				}
				else
				{
					v.texCoord = glm::vec2(i / float(fourierGridSize - 1), j / float(fourierGridSize - 1));
				}
				v.normalCoord = glm::vec3(0.0f, 1.0f, 0.0f);

				if (i - 1 >= 0 && j - 1 >= 0)
				{
					indices.push_back(i * fourierGridSize + j + fourierGridSizeX2 * timesGrid);
					indices.push_back(i * fourierGridSize + (j - 1) + fourierGridSizeX2 * timesGrid);
					indices.push_back((i - 1) * fourierGridSize + j + fourierGridSizeX2 * timesGrid);
				}

				if (i + 1 <= fourierGridSize - 1 && j + 1 <= fourierGridSize - 1)
				{
					indices.push_back(i * fourierGridSize + j + fourierGridSizeX2 * timesGrid);
					indices.push_back(i * fourierGridSize + (j + 1) + fourierGridSizeX2 * timesGrid);
					indices.push_back((i + 1) * fourierGridSize + j + fourierGridSizeX2 * timesGrid);
				}
				vertices.push_back(v);
			}
		}
		timesGrid++;

		// RIGHT

		for (int i = 0; i < fourierGridSize; i++)
		{
			for (int j = 0; j < fourierGridSize; j++)
			{
				Vertex v = {};

				v.color = waterColor;
				v.pos = glm::vec3(i - fourierGridSize + 1, 0.0f, j);
				
				if (i == fourierGridSize - 1 && j == fourierGridSize - 1)
				{
					v.texCoord = glm::vec2(0,1);
				}
				else if (i == fourierGridSize - 1 && j == 0)
				{
					v.texCoord = glm::vec2(0, 0);
				}
				else if (i == fourierGridSize - 1)
				{
					v.texCoord = glm::vec2(1 - (i / float(fourierGridSize - 1)), j / float(fourierGridSize - 1));
				}
				else if (j == 0)
				{
					v.texCoord = glm::vec2(i / float(fourierGridSize - 1), 1 - (j / float(fourierGridSize - 1)));
				}
				else
				{
					v.texCoord = glm::vec2(i / float(fourierGridSize - 1), j / float(fourierGridSize - 1));
				}
				v.normalCoord = glm::vec3(0.0f, 1.0f, 0.0f);

				if (i - 1 >= 0 && j - 1 >= 0)
				{
					indices.push_back(i * fourierGridSize + j + fourierGridSizeX2 * timesGrid);
					indices.push_back(i * fourierGridSize + (j - 1) + fourierGridSizeX2 * timesGrid);
					indices.push_back((i - 1) * fourierGridSize + j + fourierGridSizeX2 * timesGrid);
				}

				if (i + 1 <= fourierGridSize - 1 && j + 1 <= fourierGridSize - 1)
				{
					indices.push_back(i * fourierGridSize + j + fourierGridSizeX2 * timesGrid);
					indices.push_back(i * fourierGridSize + (j + 1) + fourierGridSizeX2 * timesGrid);
					indices.push_back((i + 1) * fourierGridSize + j + fourierGridSizeX2 * timesGrid);
				}
				vertices.push_back(v);
			}
		}
		timesGrid++;

		// RIGHT DOWN

		for (int i = 0; i < fourierGridSize; i++)
		{
			for (int j = 0; j < fourierGridSize; j++)
			{
				Vertex v = {};

				v.color = waterColor;
				v.pos = glm::vec3(i - fourierGridSize + 1, 0.0f, j - fourierGridSize + 1);

				if (i == fourierGridSize - 1 && j == fourierGridSize - 1)
				{
					v.texCoord = glm::vec2(0, 0);
				}
				else if (i == fourierGridSize - 1)
				{
					v.texCoord = glm::vec2(1 - (i / float(fourierGridSize - 1)), j / float(fourierGridSize - 1));
				}
				else
				{
					v.texCoord = glm::vec2(i / float(fourierGridSize - 1), j / float(fourierGridSize - 1));
				}
				v.normalCoord = glm::vec3(0.0f, 1.0f, 0.0f);

				if (i - 1 >= 0 && j - 1 >= 0)
				{
					indices.push_back(i * fourierGridSize + j + fourierGridSizeX2 * timesGrid);
					indices.push_back(i * fourierGridSize + (j - 1) + fourierGridSizeX2 * timesGrid);
					indices.push_back((i - 1) * fourierGridSize + j + fourierGridSizeX2 * timesGrid);
				}

				if (i + 1 <= fourierGridSize - 1 && j + 1 <= fourierGridSize - 1)
				{
					indices.push_back(i * fourierGridSize + j + fourierGridSizeX2 * timesGrid);
					indices.push_back(i * fourierGridSize + (j + 1) + fourierGridSizeX2 * timesGrid);
					indices.push_back((i + 1) * fourierGridSize + j + fourierGridSizeX2 * timesGrid);
				}
				vertices.push_back(v);
			}
		}
		timesGrid++;

		// DOWN

		for (int i = 0; i < fourierGridSize; i++)
		{
			for (int j = 0; j < fourierGridSize; j++)
			{
				Vertex v = {};

				v.color = waterColor;
				v.pos = glm::vec3(i, 0.0f, j - fourierGridSize + 1);
				
				if (i == fourierGridSize - 1 && j == fourierGridSize - 1)
				{
					v.texCoord = glm::vec2(1,0);
				}
				else if (i == 0 && j == fourierGridSize - 1)
				{
					v.texCoord = glm::vec2(0, 0);
				}
				else if (i == fourierGridSize - 1)
				{
					v.texCoord = glm::vec2(1-(i / float(fourierGridSize - 1)), j / float(fourierGridSize - 1));
				}
				else if (j == fourierGridSize-1)
				{
					v.texCoord = glm::vec2(i / float(fourierGridSize - 1), 1 - (j / float(fourierGridSize - 1)));
				}
				else
				{
					v.texCoord = glm::vec2(i / float(fourierGridSize - 1), j / float(fourierGridSize - 1));
				}
				
				v.normalCoord = glm::vec3(0.0f, 1.0f, 0.0f);

				if (i - 1 >= 0 && j - 1 >= 0)
				{
					indices.push_back(i * fourierGridSize + j + fourierGridSizeX2 * timesGrid);
					indices.push_back(i * fourierGridSize + (j - 1) + fourierGridSizeX2 * timesGrid);
					indices.push_back((i - 1) * fourierGridSize + j + fourierGridSizeX2 * timesGrid);
				}

				if (i + 1 <= fourierGridSize - 1 && j + 1 <= fourierGridSize - 1)
				{
					indices.push_back(i * fourierGridSize + j + fourierGridSizeX2 * timesGrid);
					indices.push_back(i * fourierGridSize + (j + 1) + fourierGridSizeX2 * timesGrid);
					indices.push_back((i + 1) * fourierGridSize + j + fourierGridSizeX2 * timesGrid);
				}
				vertices.push_back(v);
			}
		}
		timesGrid++;

		// DOWN LEFT

		for (int i = 0; i < fourierGridSize; i++)
		{
			for (int j = 0; j < fourierGridSize; j++)
			{
				Vertex v = {};

				v.color = waterColor;
				v.pos = glm::vec3(i + fourierGridSize - 1, 0.0f, j - fourierGridSize + 1);

				if (i == 0 && j == fourierGridSize - 1)
				{
					v.texCoord = glm::vec2(1, 0);
				}
				else if (j == fourierGridSize - 1)
				{
					v.texCoord = glm::vec2(i / float(fourierGridSize - 1), 1 - (j / float(fourierGridSize - 1)));
				}
				else
				{
					v.texCoord = glm::vec2(i / float(fourierGridSize - 1), j / float(fourierGridSize - 1));
				}

				v.normalCoord = glm::vec3(0.0f, 1.0f, 0.0f);

				if (i - 1 >= 0 && j - 1 >= 0)
				{
					indices.push_back(i * fourierGridSize + j + fourierGridSizeX2 * timesGrid);
					indices.push_back(i * fourierGridSize + (j - 1) + fourierGridSizeX2 * timesGrid);
					indices.push_back((i - 1) * fourierGridSize + j + fourierGridSizeX2 * timesGrid);
				}

				if (i + 1 <= fourierGridSize - 1 && j + 1 <= fourierGridSize - 1)
				{
					indices.push_back(i * fourierGridSize + j + fourierGridSizeX2 * timesGrid);
					indices.push_back(i * fourierGridSize + (j + 1) + fourierGridSizeX2 * timesGrid);
					indices.push_back((i + 1) * fourierGridSize + j + fourierGridSizeX2 * timesGrid);
				}
				vertices.push_back(v);
			}
		}
		timesGrid++;


		// QUAD
			
		indicesQuad.push_back(0);	// load the indices for the quad
		indicesQuad.push_back(1);
		indicesQuad.push_back(2);
		indicesQuad.push_back(1);
		indicesQuad.push_back(3);
		indicesQuad.push_back(2);

		// Create the 4 vertices that will create the quad
		Vertex vertex0 = {};	
		vertex0.pos = glm::vec3(-1.5, -1.5, 0.0f);
		vertex0.color = glm::vec3(1.0f, 1.0f, 1.0f);
		vertex0.texCoord = glm::vec2(0.0f, 0.0f);
		vertex0.normalCoord = glm::vec3(0.0f, 0.0f, 1.0f);

		Vertex vertex1 = {};	
		vertex1.pos = glm::vec3(-1.5,1.5, 0.0f);
		vertex1.color = glm::vec3(1.0f, 1.0f, 1.0f);
		vertex1.texCoord = glm::vec2(0.0f, 1.0f);
		vertex1.normalCoord = glm::vec3(0.0f,0.0f, 1.0f);

		Vertex vertex2 = {};	
		vertex2.pos = glm::vec3(1.5, -1.5, 0.0f);
		vertex2.color = glm::vec3(1.0f, 1.0f, 1.0f);
		vertex2.texCoord = glm::vec2(1.0f, 0.0f);
		vertex2.normalCoord = glm::vec3(0.0f, 0.0f, 1.0f);

		Vertex vertex3 = {};	
		vertex3.pos = glm::vec3(1.5, 1.5, 0.0f);
		vertex3.color = glm::vec3(1.0f, 1.0f, 1.0f);
		vertex3.texCoord = glm::vec2(1.0f, 1.0f);
		vertex3.normalCoord = glm::vec3(0.0f, 0.0f, 1.0f);
		
		// push those vertices
		verticesQuad.push_back(vertex0);
		verticesQuad.push_back(vertex1);
		verticesQuad.push_back(vertex2);
		verticesQuad.push_back(vertex3);

		// CUBE

		VertexOnlyPos vertex_0 = {};
		vertex_0.pos = glm::vec3(1.0f, 1.0f, -1.0f);

		VertexOnlyPos vertex_1 = {};
		vertex_1.pos = glm::vec3(1.0f, -1.0f, -1.0f);

		VertexOnlyPos vertex_2 = {};
		vertex_2.pos = glm::vec3(1.0f, 1.0f, 1.0f);

		VertexOnlyPos vertex_3 = {};
		vertex_3.pos = glm::vec3(1.0f, -1.0f, 1.0f);

		VertexOnlyPos vertex_4 = {};
		vertex_4.pos = glm::vec3(-1.0f, 1.0f, -1.0f);

		VertexOnlyPos vertex_5 = {};
		vertex_5.pos = glm::vec3(-1.0f, -1.0f, -1.0f);

		VertexOnlyPos vertex_6 = {};
		vertex_6.pos = glm::vec3(-1.0f, 1.0f, 1.0f);

		VertexOnlyPos vertex_7 = {};
		vertex_7.pos = glm::vec3(-1.0f, -1.0f, 1.0f);

		verticesCube.push_back(vertex_0);
		verticesCube.push_back(vertex_1);
		verticesCube.push_back(vertex_2);
		verticesCube.push_back(vertex_3);
		verticesCube.push_back(vertex_4);
		verticesCube.push_back(vertex_5);
		verticesCube.push_back(vertex_6);
		verticesCube.push_back(vertex_7);

		indicesCube.push_back(4);
		indicesCube.push_back(2);
		indicesCube.push_back(0);

		indicesCube.push_back(2);
		indicesCube.push_back(7);
		indicesCube.push_back(3);
		
		indicesCube.push_back(6);
		indicesCube.push_back(5);
		indicesCube.push_back(7);
		
		indicesCube.push_back(1);
		indicesCube.push_back(7);
		indicesCube.push_back(5);
		
		indicesCube.push_back(0);
		indicesCube.push_back(3);
		indicesCube.push_back(1);
		
		indicesCube.push_back(4);
		indicesCube.push_back(1);
		indicesCube.push_back(5);
		
		indicesCube.push_back(4);
		indicesCube.push_back(6);
		indicesCube.push_back(2);
		
		indicesCube.push_back(2);
		indicesCube.push_back(6);
		indicesCube.push_back(7);
		
		indicesCube.push_back(6);
		indicesCube.push_back(4);
		indicesCube.push_back(5);
		
		indicesCube.push_back(1);
		indicesCube.push_back(3);
		indicesCube.push_back(7);
		
		indicesCube.push_back(0);
		indicesCube.push_back(2);
		indicesCube.push_back(3);
		
		indicesCube.push_back(4);
		indicesCube.push_back(0);
		indicesCube.push_back(1);
	}
	


	// Creates a buffer to hold the information of the vertices, allocating memory and using a staging buffer
	void createVertexBuffer() {
		if (vertexBuffer)
		{
			vkDestroyBuffer(device, vertexBuffer, nullptr);
			vkFreeMemory(device, vertexBufferMemory, nullptr);
		}

		VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, vertices.data(), (size_t)bufferSize);
		vkUnmapMemory(device, stagingBufferMemory);

		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);

		copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}

	// Creates a separate buffer to hold the information of the vertices of the quad
	void createVertexBufferQuad() {
		if (vertexBufferQuad)
		{
			vkDestroyBuffer(device, vertexBufferQuad, nullptr);
			vkFreeMemory(device, vertexBufferMemoryQuad, nullptr);
		}

		VkDeviceSize bufferSize = sizeof(verticesQuad[0]) * verticesQuad.size();

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, verticesQuad.data(), (size_t)bufferSize);
		vkUnmapMemory(device, stagingBufferMemory);

		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBufferQuad, vertexBufferMemoryQuad);

		copyBuffer(stagingBuffer, vertexBufferQuad, bufferSize);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}

	// Creates a separate buffer to hold the information of the vertices of the quad
	void createVertexBufferCube() {
		if (vertexBufferCube)
		{
			vkDestroyBuffer(device, vertexBufferCube, nullptr);
			vkFreeMemory(device, vertexBufferMemoryCube, nullptr);
		}

		VkDeviceSize bufferSize = sizeof(verticesCube[0]) * verticesCube.size();

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, verticesCube.data(), (size_t)bufferSize);
		vkUnmapMemory(device, stagingBufferMemory);

		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBufferCube, vertexBufferMemoryCube);

		copyBuffer(stagingBuffer, vertexBufferCube, bufferSize);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}

	// Creates a buffer to hold the indices of the vertices, allocating memory and using a staging buffer
	void createIndexBuffer() {
		if (indexBuffer)
		{
			vkDestroyBuffer(device, indexBuffer, nullptr);
			vkFreeMemory(device, indexBufferMemory, nullptr);
		}
		VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, indices.data(), (size_t)bufferSize);
		vkUnmapMemory(device, stagingBufferMemory);

		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);

		copyBuffer(stagingBuffer, indexBuffer, bufferSize);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}

	// Creates a separate buffer to hold the indices of the vertices of the quad
	void createIndexBufferQuad() {
		if (indexBufferQuad)
		{
			vkDestroyBuffer(device, indexBufferQuad, nullptr);
			vkFreeMemory(device, indexBufferMemoryQuad, nullptr);
		}

		VkDeviceSize bufferSize = sizeof(indicesQuad[0]) * indicesQuad.size();

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, indicesQuad.data(), (size_t)bufferSize);
		vkUnmapMemory(device, stagingBufferMemory);

		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBufferQuad, indexBufferMemoryQuad);

		copyBuffer(stagingBuffer, indexBufferQuad, bufferSize);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}

	// Creates a separate buffer to hold the indices of the vertices of the quad
	void createIndexBufferCube() {
		if (indexBufferCube)
		{
			vkDestroyBuffer(device, indexBufferCube, nullptr);
			vkFreeMemory(device, indexBufferMemoryCube, nullptr);
		}

		VkDeviceSize bufferSize = sizeof(indicesCube[0]) * indicesCube.size();

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, indicesCube.data(), (size_t)bufferSize);
		vkUnmapMemory(device, stagingBufferMemory);

		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBufferCube, indexBufferMemoryCube);

		copyBuffer(stagingBuffer, indexBufferCube, bufferSize);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}

	// Create the uniform buffer for the object one per image in the swap chain
	void createUniformBuffers() {
		VkDeviceSize bufferSize = sizeof(UniformBufferObject);

		uniformBuffers.resize(swapChainImages.size());
		uniformBuffersMemory.resize(swapChainImages.size());

		for (size_t i = 0; i < swapChainImages.size(); i++) {
			createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers[i], uniformBuffersMemory[i]);
		}
	}

	// Create the uniform buffer for the light one per image in the swap chain
	void createLightUniformBuffers() {
		VkDeviceSize bufferSize = sizeof(UniformBufferLight);

		lightUniformBuffers.resize(swapChainImages.size());
		lightUniformBuffersMemory.resize(swapChainImages.size());

		for (size_t i = 0; i < swapChainImages.size(); i++) {
			createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, lightUniformBuffers[i], lightUniformBuffersMemory[i]);
		}
	}

	// Create the uniform buffer for the light one per image in the swap chain
	void creategaussianNoiseStorageBuffers() {

		std::vector<StorageBufferGaussianNoise> gaussianNoiseBuffer(fourierGridSize * fourierGridSize);
		for (auto& gaussianNoise : gaussianNoiseBuffer) {
			gaussianNoise.gaussianNoise = glm::vec4(distribution(generator), distribution(generator), distribution(generator), distribution(generator));
		}

		VkDeviceSize bufferSize = gaussianNoiseBuffer.size() * sizeof(StorageBufferGaussianNoise);

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, gaussianNoiseBuffer.data(), (size_t)bufferSize);
		vkUnmapMemory(device, stagingBufferMemory);

		createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, gaussianNoiseStorageBuffers, gaussianNoiseStorageBuffersMemory);

		copyBuffer(stagingBuffer, gaussianNoiseStorageBuffers, bufferSize);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}

	// Create the uniform buffer for the light one per image in the swap chain
	void createIndicesFFTAuxStorageBuffers() {
		int log_2_N = int(log2(fourierGridSize));
		std::vector<StorageBufferFFTAux> FFTAuxBuffer;

		for (int k = 0; k < fourierGridSize; k++) {
			
			int i = k;
			StorageBufferFFTAux res;
			res.index = 0;
			for (int j = 0; j < log_2_N; j++) {
				res.index = (res.index << 1) + (i & 1);
				i >>= 1;
			}
			
			FFTAuxBuffer.push_back(res);
			//std::cout << res.index << " " << k << std::endl;
		}

		VkDeviceSize bufferSize = FFTAuxBuffer.size() * sizeof(StorageBufferFFTAux);

		VkBuffer stagingBuffer;
		VkDeviceMemory stagingBufferMemory;
		createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

		void* data;
		vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
		memcpy(data, FFTAuxBuffer.data(), (size_t)bufferSize);
		vkUnmapMemory(device, stagingBufferMemory);

		createBuffer(bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, FFTAuxStorageBuffers, FFTAuxStorageBuffersMemory);

		copyBuffer(stagingBuffer, FFTAuxStorageBuffers, bufferSize);

		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingBufferMemory, nullptr);
	}

	// Create the uniform buffer for the object one per image in the swap chain
	void createUniformBuffersTimeHkt() {
		VkDeviceSize bufferSize = sizeof(UniformBufferObjectTimeHkt);

		uniformBuffersTimeHkt.resize(swapChainImages.size());
		uniformBuffersTimeHktMemory.resize(swapChainImages.size());

		for (size_t i = 0; i < swapChainImages.size(); i++) {
			createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffersTimeHkt[i], uniformBuffersTimeHktMemory[i]);
		}
	}

	// Creates a descriptor pool holding information for the Gui
	void createDescriptorPoolGUI() {
		std::array<VkDescriptorPoolSize, 11> poolSizes = {};
		poolSizes[0] = { VK_DESCRIPTOR_TYPE_SAMPLER, 1000 };
		poolSizes[1] = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 };
		poolSizes[2] = { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 };
		poolSizes[3] = { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 };
		poolSizes[4] = { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 };
		poolSizes[5] = { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 };
		poolSizes[6] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 };
		poolSizes[7] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 };
		poolSizes[8] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 };
		poolSizes[9] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 };
		poolSizes[10] = { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 };
		

		VkDescriptorPoolCreateInfo poolInfo = {};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
		poolInfo.pPoolSizes = poolSizes.data();
		poolInfo.maxSets = static_cast<uint32_t>(swapChainImages.size());

		if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPoolGUI) != VK_SUCCESS) {
			throw std::runtime_error("failed to create descriptor pool!");
		}
	}

	// Creates a descriptor pool holding all the information of the buffers that need to be transfered to the GPU
	void createDescriptorPool() {
		std::array<VkDescriptorPoolSize, 47> poolSizes = {};
		poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		poolSizes[0].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
		poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSizes[1].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
		poolSizes[2].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		poolSizes[2].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
		poolSizes[3].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		poolSizes[3].descriptorCount = 1;
		poolSizes[4].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		poolSizes[4].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
		poolSizes[5].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSizes[5].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
		poolSizes[6].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		poolSizes[6].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
		poolSizes[7].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSizes[7].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
		poolSizes[8].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		poolSizes[8].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
		poolSizes[9].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSizes[9].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
		poolSizes[10].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		poolSizes[10].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
		poolSizes[11].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		poolSizes[11].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
		poolSizes[12].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSizes[12].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
		poolSizes[13].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		poolSizes[13].descriptorCount = 1;
		poolSizes[14].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		poolSizes[14].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
		poolSizes[15].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSizes[15].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
		poolSizes[16].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		poolSizes[16].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
		poolSizes[17].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSizes[17].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
		poolSizes[18].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSizes[18].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
		poolSizes[19].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		poolSizes[19].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
		poolSizes[20].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSizes[20].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
		poolSizes[21].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		poolSizes[21].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
		poolSizes[22].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSizes[22].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
		poolSizes[23].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		poolSizes[23].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
		poolSizes[24].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSizes[24].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
		poolSizes[25].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		poolSizes[25].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
		poolSizes[26].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSizes[26].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
		poolSizes[27].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		poolSizes[27].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
		poolSizes[28].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSizes[28].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
		poolSizes[29].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		poolSizes[29].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
		poolSizes[30].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSizes[30].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
		poolSizes[31].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		poolSizes[31].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
		poolSizes[32].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSizes[32].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
		poolSizes[33].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		poolSizes[33].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
		poolSizes[34].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSizes[34].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
		poolSizes[35].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		poolSizes[35].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
		poolSizes[36].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSizes[36].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
		poolSizes[37].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSizes[37].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
		poolSizes[38].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		poolSizes[38].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
		poolSizes[39].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSizes[39].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
		poolSizes[40].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSizes[40].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
		poolSizes[41].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		poolSizes[41].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
		poolSizes[42].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSizes[42].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
		poolSizes[43].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSizes[43].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
		poolSizes[44].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		poolSizes[44].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
		poolSizes[45].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSizes[45].descriptorCount = static_cast<uint32_t>(swapChainImages.size());
		poolSizes[46].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		poolSizes[46].descriptorCount = static_cast<uint32_t>(swapChainImages.size());

		VkDescriptorPoolCreateInfo poolInfo = {};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
		poolInfo.pPoolSizes = poolSizes.data();
		poolInfo.maxSets = static_cast<uint32_t>(swapChainImages.size());

		if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
			throw std::runtime_error("failed to create descriptor pool!");
		}
	}

	// Holds information of the different descriptors that will be used
	void createDescriptorSets() {
		std::vector<VkDescriptorSetLayout> layouts(swapChainImages.size(), descriptorSetLayout);
		VkDescriptorSetAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = descriptorPool;
		allocInfo.descriptorSetCount = static_cast<uint32_t>(swapChainImages.size());
		allocInfo.pSetLayouts = layouts.data();

		descriptorSets.resize(swapChainImages.size());
		if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate descriptor sets!");
		}

		for (size_t i = 0; i < swapChainImages.size(); i++) {
			VkDescriptorBufferInfo bufferInfo = {};
			bufferInfo.buffer = uniformBuffers[i];
			bufferInfo.offset = 0;
			bufferInfo.range = sizeof(UniformBufferObject);

			VkDescriptorImageInfo imageInfo = {};
			imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			imageInfo.imageView = textureImageView;
			imageInfo.sampler = textureSamplerCube;

			VkDescriptorBufferInfo bufferLightInfo = {};
			bufferLightInfo.buffer = lightUniformBuffers[i];
			bufferLightInfo.offset = 0;
			bufferLightInfo.range = sizeof(UniformBufferLight);

			VkDescriptorBufferInfo bufferGaussianNoiseInfo = {};
			bufferGaussianNoiseInfo.buffer = gaussianNoiseStorageBuffers;
			bufferGaussianNoiseInfo.offset = 0;
			bufferGaussianNoiseInfo.range = sizeof(StorageBufferGaussianNoise)*fourierGridSize*fourierGridSize;

			VkDescriptorImageInfo computeShaderImageInfoHk0 = {};
			computeShaderImageInfoHk0.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
			computeShaderImageInfoHk0.imageView = computeShaderImageViewHk0;
			computeShaderImageInfoHk0.sampler = textureSampler;

			VkDescriptorImageInfo computeShaderImageInfoHk0minus = {};
			computeShaderImageInfoHk0minus.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
			computeShaderImageInfoHk0minus.imageView = computeShaderImageViewHk0minus;
			computeShaderImageInfoHk0minus.sampler = textureSampler;

			VkDescriptorImageInfo computeShaderImageInfoHkt = {};
			computeShaderImageInfoHkt.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
			computeShaderImageInfoHkt.imageView = computeShaderImageViewHkt;
			computeShaderImageInfoHkt.sampler = textureSampler;

			VkDescriptorBufferInfo bufferHktInfo = {};
			bufferHktInfo.buffer = uniformBuffersTimeHkt[i];
			bufferHktInfo.offset = 0;
			bufferHktInfo.range = sizeof(UniformBufferObjectTimeHkt);

			VkDescriptorImageInfo computeShaderImageInfoFFTAux = {};
			computeShaderImageInfoFFTAux.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
			computeShaderImageInfoFFTAux.imageView = computeShaderImageViewFFTAux;
			computeShaderImageInfoFFTAux.sampler = textureSampler;

			VkDescriptorBufferInfo bufferFFTAux = {};
			bufferFFTAux.buffer = FFTAuxStorageBuffers;
			bufferFFTAux.offset = 0;
			bufferFFTAux.range = sizeof(StorageBufferFFTAux)*fourierGridSize;

			VkDescriptorImageInfo computeShaderImageInfoFFTAlternate = {};
			computeShaderImageInfoFFTAlternate.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
			computeShaderImageInfoFFTAlternate.imageView = computeShaderImageViewFFTAlternate;
			computeShaderImageInfoFFTAlternate.sampler = textureSampler;

			VkDescriptorImageInfo computeShaderImageInfoHeightMap = {};
			computeShaderImageInfoHeightMap.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
			computeShaderImageInfoHeightMap.imageView = computeShaderImageViewHeightMap;
			computeShaderImageInfoHeightMap.sampler = textureSampler;

			VkDescriptorImageInfo computeShaderImageInfoHeightMapVertex = {};
			computeShaderImageInfoHeightMapVertex.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
			computeShaderImageInfoHeightMapVertex.imageView = computeShaderImageViewHeightMap;
			computeShaderImageInfoHeightMapVertex.sampler = textureSampler;

			VkDescriptorImageInfo computeShaderImageInfoSlopeX = {};
			computeShaderImageInfoSlopeX.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
			computeShaderImageInfoSlopeX.imageView = computeShaderImageViewSlopeX;
			computeShaderImageInfoSlopeX.sampler = textureSampler;

			VkDescriptorImageInfo computeShaderImageInfoSlopeXAlternate = {};
			computeShaderImageInfoSlopeXAlternate.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
			computeShaderImageInfoSlopeXAlternate.imageView = computeShaderImageViewSlopeXAlternate;
			computeShaderImageInfoSlopeXAlternate.sampler = textureSampler;

			VkDescriptorImageInfo computeShaderImageInfoSlopeZ = {};
			computeShaderImageInfoSlopeZ.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
			computeShaderImageInfoSlopeZ.imageView = computeShaderImageViewSlopeZ;
			computeShaderImageInfoSlopeZ.sampler = textureSampler;

			VkDescriptorImageInfo computeShaderImageInfoSlopeZAlternate = {};
			computeShaderImageInfoSlopeZAlternate.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
			computeShaderImageInfoSlopeZAlternate.imageView = computeShaderImageViewSlopeZAlternate;
			computeShaderImageInfoSlopeZAlternate.sampler = textureSampler;

			VkDescriptorImageInfo computeShaderImageInfoDispX = {};
			computeShaderImageInfoDispX.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
			computeShaderImageInfoDispX.imageView = computeShaderImageViewDispX;
			computeShaderImageInfoDispX.sampler = textureSampler;

			VkDescriptorImageInfo computeShaderImageInfoDispXAlternate = {};
			computeShaderImageInfoDispXAlternate.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
			computeShaderImageInfoDispXAlternate.imageView = computeShaderImageViewDispXAlternate;
			computeShaderImageInfoDispXAlternate.sampler = textureSampler;

			VkDescriptorImageInfo computeShaderImageInfoDispZ = {};
			computeShaderImageInfoDispZ.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
			computeShaderImageInfoDispZ.imageView = computeShaderImageViewDispZ;
			computeShaderImageInfoDispZ.sampler = textureSampler;

			VkDescriptorImageInfo computeShaderImageInfoDispZAlternate = {};
			computeShaderImageInfoDispZAlternate.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
			computeShaderImageInfoDispZAlternate.imageView = computeShaderImageViewDispZAlternate;
			computeShaderImageInfoDispZAlternate.sampler = textureSampler;

			VkDescriptorImageInfo computeShaderImageInfoSlopeXFinal = {};
			computeShaderImageInfoSlopeXFinal.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
			computeShaderImageInfoSlopeXFinal.imageView = computeShaderImageViewSlopeXFinal;
			computeShaderImageInfoSlopeXFinal.sampler = textureSampler;

			VkDescriptorImageInfo computeShaderImageInfoSlopeXVertex = {};
			computeShaderImageInfoSlopeXVertex.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
			computeShaderImageInfoSlopeXVertex.imageView = computeShaderImageViewSlopeXFinal;
			computeShaderImageInfoSlopeXVertex.sampler = textureSampler;

			VkDescriptorImageInfo computeShaderImageInfoSlopeZFinal = {};
			computeShaderImageInfoSlopeZFinal.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
			computeShaderImageInfoSlopeZFinal.imageView = computeShaderImageViewSlopeZFinal;
			computeShaderImageInfoSlopeZFinal.sampler = textureSampler;

			VkDescriptorImageInfo computeShaderImageInfoSlopeZVertex = {};
			computeShaderImageInfoSlopeZVertex.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
			computeShaderImageInfoSlopeZVertex.imageView = computeShaderImageViewSlopeZFinal;
			computeShaderImageInfoSlopeZVertex.sampler = textureSampler;

			VkDescriptorImageInfo computeShaderImageInfoDispXFinal = {};
			computeShaderImageInfoDispXFinal.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
			computeShaderImageInfoDispXFinal.imageView = computeShaderImageViewDispXFinal;
			computeShaderImageInfoDispXFinal.sampler = textureSampler;

			VkDescriptorImageInfo computeShaderImageInfoDispXVertex = {};
			computeShaderImageInfoDispXVertex.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
			computeShaderImageInfoDispXVertex.imageView = computeShaderImageViewDispXFinal;
			computeShaderImageInfoDispXVertex.sampler = textureSampler;

			VkDescriptorImageInfo computeShaderImageInfoDispZFinal = {};
			computeShaderImageInfoDispZFinal.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
			computeShaderImageInfoDispZFinal.imageView = computeShaderImageViewDispZFinal;
			computeShaderImageInfoDispZFinal.sampler = textureSampler;

			VkDescriptorImageInfo computeShaderImageInfoDispZVertex = {};
			computeShaderImageInfoDispZVertex.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
			computeShaderImageInfoDispZVertex.imageView = computeShaderImageViewDispZFinal;
			computeShaderImageInfoDispZVertex.sampler = textureSampler;

			std::array<VkWriteDescriptorSet, 47> descriptorWrites = {};

			descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[0].dstSet = descriptorSets[i];
			descriptorWrites[0].dstBinding = 0;
			descriptorWrites[0].dstArrayElement = 0;
			descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			descriptorWrites[0].descriptorCount = 1;
			descriptorWrites[0].pBufferInfo = &bufferInfo;

			descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[1].dstSet = descriptorSets[i];
			descriptorWrites[1].dstBinding = 1;
			descriptorWrites[1].dstArrayElement = 0;
			descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descriptorWrites[1].descriptorCount = 1;
			descriptorWrites[1].pImageInfo = &imageInfo;

			descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[2].dstSet = descriptorSets[i];
			descriptorWrites[2].dstBinding = 2;
			descriptorWrites[2].dstArrayElement = 0;
			descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			descriptorWrites[2].descriptorCount = 1;
			descriptorWrites[2].pBufferInfo = &bufferLightInfo;

			descriptorWrites[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[3].dstSet = descriptorSets[i];
			descriptorWrites[3].dstBinding = 3;
			descriptorWrites[3].dstArrayElement = 0;
			descriptorWrites[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			descriptorWrites[3].descriptorCount = 1;
			descriptorWrites[3].pBufferInfo = &bufferGaussianNoiseInfo;

			descriptorWrites[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[4].dstSet = descriptorSets[i];
			descriptorWrites[4].dstBinding = 4;
			descriptorWrites[4].dstArrayElement = 0;
			descriptorWrites[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
			descriptorWrites[4].descriptorCount = 1;
			descriptorWrites[4].pImageInfo = &computeShaderImageInfoHk0;
			
			descriptorWrites[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[5].dstSet = descriptorSets[i];
			descriptorWrites[5].dstBinding = 5;
			descriptorWrites[5].dstArrayElement = 0;
			descriptorWrites[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descriptorWrites[5].descriptorCount = 1;
			descriptorWrites[5].pImageInfo = &computeShaderImageInfoHk0;

			descriptorWrites[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[6].dstSet = descriptorSets[i];
			descriptorWrites[6].dstBinding = 6;
			descriptorWrites[6].dstArrayElement = 0;
			descriptorWrites[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
			descriptorWrites[6].descriptorCount = 1;
			descriptorWrites[6].pImageInfo = &computeShaderImageInfoHk0minus;

			descriptorWrites[7].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[7].dstSet = descriptorSets[i];
			descriptorWrites[7].dstBinding = 7;
			descriptorWrites[7].dstArrayElement = 0;
			descriptorWrites[7].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descriptorWrites[7].descriptorCount = 1;
			descriptorWrites[7].pImageInfo = &computeShaderImageInfoHk0minus;

			descriptorWrites[8].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[8].dstSet = descriptorSets[i];
			descriptorWrites[8].dstBinding = 8;
			descriptorWrites[8].dstArrayElement = 0;
			descriptorWrites[8].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
			descriptorWrites[8].descriptorCount = 1;
			descriptorWrites[8].pImageInfo = &computeShaderImageInfoHkt;

			descriptorWrites[9].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[9].dstSet = descriptorSets[i];
			descriptorWrites[9].dstBinding = 9;
			descriptorWrites[9].dstArrayElement = 0;
			descriptorWrites[9].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descriptorWrites[9].descriptorCount = 1;
			descriptorWrites[9].pImageInfo = &computeShaderImageInfoHkt;

			descriptorWrites[10].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[10].dstSet = descriptorSets[i];
			descriptorWrites[10].dstBinding = 10;
			descriptorWrites[10].dstArrayElement = 0;
			descriptorWrites[10].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			descriptorWrites[10].descriptorCount = 1;
			descriptorWrites[10].pBufferInfo = &bufferHktInfo;

			descriptorWrites[11].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[11].dstSet = descriptorSets[i];
			descriptorWrites[11].dstBinding = 11;
			descriptorWrites[11].dstArrayElement = 0;
			descriptorWrites[11].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
			descriptorWrites[11].descriptorCount = 1;
			descriptorWrites[11].pImageInfo = &computeShaderImageInfoFFTAux;

			descriptorWrites[12].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[12].dstSet = descriptorSets[i];
			descriptorWrites[12].dstBinding = 12;
			descriptorWrites[12].dstArrayElement = 0;
			descriptorWrites[12].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descriptorWrites[12].descriptorCount = 1;
			descriptorWrites[12].pImageInfo = &computeShaderImageInfoFFTAux;

			descriptorWrites[13].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[13].dstSet = descriptorSets[i];
			descriptorWrites[13].dstBinding = 13;
			descriptorWrites[13].dstArrayElement = 0;
			descriptorWrites[13].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			descriptorWrites[13].descriptorCount = 1;
			descriptorWrites[13].pBufferInfo = &bufferFFTAux;

			descriptorWrites[14].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[14].dstSet = descriptorSets[i];
			descriptorWrites[14].dstBinding = 14;
			descriptorWrites[14].dstArrayElement = 0;
			descriptorWrites[14].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
			descriptorWrites[14].descriptorCount = 1;
			descriptorWrites[14].pImageInfo = &computeShaderImageInfoFFTAlternate;

			descriptorWrites[15].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[15].dstSet = descriptorSets[i];
			descriptorWrites[15].dstBinding = 15;
			descriptorWrites[15].dstArrayElement = 0;
			descriptorWrites[15].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descriptorWrites[15].descriptorCount = 1;
			descriptorWrites[15].pImageInfo = &computeShaderImageInfoFFTAlternate;

			descriptorWrites[16].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[16].dstSet = descriptorSets[i];
			descriptorWrites[16].dstBinding = 16;
			descriptorWrites[16].dstArrayElement = 0;
			descriptorWrites[16].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
			descriptorWrites[16].descriptorCount = 1;
			descriptorWrites[16].pImageInfo = &computeShaderImageInfoHeightMap;

			descriptorWrites[17].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[17].dstSet = descriptorSets[i];
			descriptorWrites[17].dstBinding = 17;
			descriptorWrites[17].dstArrayElement = 0;
			descriptorWrites[17].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descriptorWrites[17].descriptorCount = 1;
			descriptorWrites[17].pImageInfo = &computeShaderImageInfoHeightMap;

			descriptorWrites[18].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[18].dstSet = descriptorSets[i];
			descriptorWrites[18].dstBinding = 18;
			descriptorWrites[18].dstArrayElement = 0;
			descriptorWrites[18].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descriptorWrites[18].descriptorCount = 1;
			descriptorWrites[18].pImageInfo = &computeShaderImageInfoHeightMapVertex;

			descriptorWrites[19].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[19].dstSet = descriptorSets[i];
			descriptorWrites[19].dstBinding = 19;
			descriptorWrites[19].dstArrayElement = 0;
			descriptorWrites[19].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
			descriptorWrites[19].descriptorCount = 1;
			descriptorWrites[19].pImageInfo = &computeShaderImageInfoSlopeX;

			descriptorWrites[20].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[20].dstSet = descriptorSets[i];
			descriptorWrites[20].dstBinding = 20;
			descriptorWrites[20].dstArrayElement = 0;
			descriptorWrites[20].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descriptorWrites[20].descriptorCount = 1;
			descriptorWrites[20].pImageInfo = &computeShaderImageInfoSlopeX;

			descriptorWrites[21].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[21].dstSet = descriptorSets[i];
			descriptorWrites[21].dstBinding = 21;
			descriptorWrites[21].dstArrayElement = 0;
			descriptorWrites[21].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
			descriptorWrites[21].descriptorCount = 1;
			descriptorWrites[21].pImageInfo = &computeShaderImageInfoSlopeXAlternate;

			descriptorWrites[22].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[22].dstSet = descriptorSets[i];
			descriptorWrites[22].dstBinding = 22;
			descriptorWrites[22].dstArrayElement = 0;
			descriptorWrites[22].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descriptorWrites[22].descriptorCount = 1;
			descriptorWrites[22].pImageInfo = &computeShaderImageInfoSlopeXAlternate;

			descriptorWrites[23].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[23].dstSet = descriptorSets[i];
			descriptorWrites[23].dstBinding = 23;
			descriptorWrites[23].dstArrayElement = 0;
			descriptorWrites[23].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
			descriptorWrites[23].descriptorCount = 1;
			descriptorWrites[23].pImageInfo = &computeShaderImageInfoSlopeZ;

			descriptorWrites[24].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[24].dstSet = descriptorSets[i];
			descriptorWrites[24].dstBinding = 24;
			descriptorWrites[24].dstArrayElement = 0;
			descriptorWrites[24].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descriptorWrites[24].descriptorCount = 1;
			descriptorWrites[24].pImageInfo = &computeShaderImageInfoSlopeZ;

			descriptorWrites[25].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[25].dstSet = descriptorSets[i];
			descriptorWrites[25].dstBinding = 25;
			descriptorWrites[25].dstArrayElement = 0;
			descriptorWrites[25].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
			descriptorWrites[25].descriptorCount = 1;
			descriptorWrites[25].pImageInfo = &computeShaderImageInfoSlopeZAlternate;

			descriptorWrites[26].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[26].dstSet = descriptorSets[i];
			descriptorWrites[26].dstBinding = 26;
			descriptorWrites[26].dstArrayElement = 0;
			descriptorWrites[26].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descriptorWrites[26].descriptorCount = 1;
			descriptorWrites[26].pImageInfo = &computeShaderImageInfoSlopeZAlternate;

			descriptorWrites[27].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[27].dstSet = descriptorSets[i];
			descriptorWrites[27].dstBinding = 27;
			descriptorWrites[27].dstArrayElement = 0;
			descriptorWrites[27].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
			descriptorWrites[27].descriptorCount = 1;
			descriptorWrites[27].pImageInfo = &computeShaderImageInfoDispX;

			descriptorWrites[28].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[28].dstSet = descriptorSets[i];
			descriptorWrites[28].dstBinding = 28;
			descriptorWrites[28].dstArrayElement = 0;
			descriptorWrites[28].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descriptorWrites[28].descriptorCount = 1;
			descriptorWrites[28].pImageInfo = &computeShaderImageInfoDispX;

			descriptorWrites[29].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[29].dstSet = descriptorSets[i];
			descriptorWrites[29].dstBinding = 29;
			descriptorWrites[29].dstArrayElement = 0;
			descriptorWrites[29].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
			descriptorWrites[29].descriptorCount = 1;
			descriptorWrites[29].pImageInfo = &computeShaderImageInfoDispXAlternate;

			descriptorWrites[30].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[30].dstSet = descriptorSets[i];
			descriptorWrites[30].dstBinding = 30;
			descriptorWrites[30].dstArrayElement = 0;
			descriptorWrites[30].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descriptorWrites[30].descriptorCount = 1;
			descriptorWrites[30].pImageInfo = &computeShaderImageInfoDispXAlternate;

			descriptorWrites[31].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[31].dstSet = descriptorSets[i];
			descriptorWrites[31].dstBinding = 31;
			descriptorWrites[31].dstArrayElement = 0;
			descriptorWrites[31].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
			descriptorWrites[31].descriptorCount = 1;
			descriptorWrites[31].pImageInfo = &computeShaderImageInfoDispZ;

			descriptorWrites[32].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[32].dstSet = descriptorSets[i];
			descriptorWrites[32].dstBinding = 32;
			descriptorWrites[32].dstArrayElement = 0;
			descriptorWrites[32].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descriptorWrites[32].descriptorCount = 1;
			descriptorWrites[32].pImageInfo = &computeShaderImageInfoDispZ;

			descriptorWrites[33].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[33].dstSet = descriptorSets[i];
			descriptorWrites[33].dstBinding = 33;
			descriptorWrites[33].dstArrayElement = 0;
			descriptorWrites[33].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
			descriptorWrites[33].descriptorCount = 1;
			descriptorWrites[33].pImageInfo = &computeShaderImageInfoDispZAlternate;

			descriptorWrites[34].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[34].dstSet = descriptorSets[i];
			descriptorWrites[34].dstBinding = 34;
			descriptorWrites[34].dstArrayElement = 0;
			descriptorWrites[34].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descriptorWrites[34].descriptorCount = 1;
			descriptorWrites[34].pImageInfo = &computeShaderImageInfoDispZAlternate;

			descriptorWrites[35].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[35].dstSet = descriptorSets[i];
			descriptorWrites[35].dstBinding = 35;
			descriptorWrites[35].dstArrayElement = 0;
			descriptorWrites[35].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
			descriptorWrites[35].descriptorCount = 1;
			descriptorWrites[35].pImageInfo = &computeShaderImageInfoSlopeXFinal;

			descriptorWrites[36].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[36].dstSet = descriptorSets[i];
			descriptorWrites[36].dstBinding = 36;
			descriptorWrites[36].dstArrayElement = 0;
			descriptorWrites[36].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descriptorWrites[36].descriptorCount = 1;
			descriptorWrites[36].pImageInfo = &computeShaderImageInfoSlopeXFinal;

			descriptorWrites[37].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[37].dstSet = descriptorSets[i];
			descriptorWrites[37].dstBinding = 37;
			descriptorWrites[37].dstArrayElement = 0;
			descriptorWrites[37].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descriptorWrites[37].descriptorCount = 1;
			descriptorWrites[37].pImageInfo = &computeShaderImageInfoSlopeXVertex;

			descriptorWrites[38].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[38].dstSet = descriptorSets[i];
			descriptorWrites[38].dstBinding = 38;
			descriptorWrites[38].dstArrayElement = 0;
			descriptorWrites[38].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
			descriptorWrites[38].descriptorCount = 1;
			descriptorWrites[38].pImageInfo = &computeShaderImageInfoSlopeZFinal;

			descriptorWrites[39].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[39].dstSet = descriptorSets[i];
			descriptorWrites[39].dstBinding = 39;
			descriptorWrites[39].dstArrayElement = 0;
			descriptorWrites[39].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descriptorWrites[39].descriptorCount = 1;
			descriptorWrites[39].pImageInfo = &computeShaderImageInfoSlopeZFinal;

			descriptorWrites[40].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[40].dstSet = descriptorSets[i];
			descriptorWrites[40].dstBinding = 40;
			descriptorWrites[40].dstArrayElement = 0;
			descriptorWrites[40].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descriptorWrites[40].descriptorCount = 1;
			descriptorWrites[40].pImageInfo = &computeShaderImageInfoSlopeZVertex;

			descriptorWrites[41].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[41].dstSet = descriptorSets[i];
			descriptorWrites[41].dstBinding = 41;
			descriptorWrites[41].dstArrayElement = 0;
			descriptorWrites[41].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
			descriptorWrites[41].descriptorCount = 1;
			descriptorWrites[41].pImageInfo = &computeShaderImageInfoDispXFinal;

			descriptorWrites[42].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[42].dstSet = descriptorSets[i];
			descriptorWrites[42].dstBinding = 42;
			descriptorWrites[42].dstArrayElement = 0;
			descriptorWrites[42].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descriptorWrites[42].descriptorCount = 1;
			descriptorWrites[42].pImageInfo = &computeShaderImageInfoDispXFinal;

			descriptorWrites[43].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[43].dstSet = descriptorSets[i];
			descriptorWrites[43].dstBinding = 43;
			descriptorWrites[43].dstArrayElement = 0;
			descriptorWrites[43].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descriptorWrites[43].descriptorCount = 1;
			descriptorWrites[43].pImageInfo = &computeShaderImageInfoDispXVertex;

			descriptorWrites[44].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[44].dstSet = descriptorSets[i];
			descriptorWrites[44].dstBinding = 44;
			descriptorWrites[44].dstArrayElement = 0;
			descriptorWrites[44].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
			descriptorWrites[44].descriptorCount = 1;
			descriptorWrites[44].pImageInfo = &computeShaderImageInfoDispZFinal;

			descriptorWrites[45].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[45].dstSet = descriptorSets[i];
			descriptorWrites[45].dstBinding = 45;
			descriptorWrites[45].dstArrayElement = 0;
			descriptorWrites[45].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descriptorWrites[45].descriptorCount = 1;
			descriptorWrites[45].pImageInfo = &computeShaderImageInfoDispZFinal;

			descriptorWrites[46].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[46].dstSet = descriptorSets[i];
			descriptorWrites[46].dstBinding = 46;
			descriptorWrites[46].dstArrayElement = 0;
			descriptorWrites[46].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descriptorWrites[46].descriptorCount = 1;
			descriptorWrites[46].pImageInfo = &computeShaderImageInfoDispZVertex;

			vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
		}
	}

	// General method to create any kind of buffer allocating memory and binding it
	void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
		VkBufferCreateInfo bufferInfo = {};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = size;
		bufferInfo.usage = usage;
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
			throw std::runtime_error("failed to create buffer!");
		}

		VkMemoryRequirements memRequirements;
		vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

		VkMemoryAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

		if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate buffer memory!");
		}

		vkBindBufferMemory(device, buffer, bufferMemory, 0);
	}

	// Auxiliar method for executing command buffers to be called at the beginning of the execution
	VkCommandBuffer beginSingleTimeCommands() {
		VkCommandBufferAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandPool = commandPool;
		allocInfo.commandBufferCount = 1;

		VkCommandBuffer commandBuffer;
		vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

		VkCommandBufferBeginInfo beginInfo = {};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		vkBeginCommandBuffer(commandBuffer, &beginInfo);

		return commandBuffer;
	}

	// Auxiliar method for executing command buffers to be called at the end of the execution
	void endSingleTimeCommands(VkCommandBuffer commandBuffer) {
		vkEndCommandBuffer(commandBuffer);

		VkSubmitInfo submitInfo = {};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;

		vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
		vkQueueWaitIdle(graphicsQueue);

		vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
	}

	// Performs the copy of a source buffer in a destination buffer
	void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkBufferCopy copyRegion = {};
		copyRegion.size = size;
		vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

		endSingleTimeCommands(commandBuffer);
	}

	uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
		VkPhysicalDeviceMemoryProperties memProperties;
		vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

		for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
			if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
				return i;
			}
		}

		throw std::runtime_error("failed to find suitable memory type!");
	}

	// Creates a command buffer per image in the swap chain to be able to bind the correct buffer during the draw call
	void createPreprocessCommandBuffers() {
		VkCommandBuffer command_buffer = beginSingleTimeCommands();
	
		// Create the H0k and H0kminus textures
		{
			pushConstantsComputeH0kPipeline pcsp = {};
			pcsp.fourierGridSize = fourierGridSize;
			pcsp.spatialDimension = spatialDimension;
			pcsp.windDirection = windDirection;
			pcsp.windSpeed = windSpeed;
			pcsp.scalePhillips = scalePhillips;

			vkCmdPushConstants(
				command_buffer,
				computeH0kPipelineLayout,
				VK_SHADER_STAGE_COMPUTE_BIT,
				0,
				sizeof(pushConstantsComputeH0kPipeline),
				&pcsp);

			vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, computeH0kPipeline);
			vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, computeH0kPipelineLayout, 0, 1, &descriptorSets[0], 0, nullptr);

			vkCmdDispatch(command_buffer, fourierGridSize / 32, fourierGridSize / 32, 1);
		}
		


		// Create the FFT Auxiliar texture
		{
			pushConstantsComputeFFTAuxPipeline pcsp = {};
			pcsp.fourierGridSize = fourierGridSize;

			vkCmdPushConstants(
				command_buffer,
				computeFFTAuxPipelineLayout,
				VK_SHADER_STAGE_COMPUTE_BIT,
				0,
				sizeof(pushConstantsComputeFFTAuxPipeline),
				&pcsp);

			vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, computeFFTAuxPipeline);
			vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, computeFFTAuxPipelineLayout, 0, 1, &descriptorSets[0], 0, nullptr);

			vkCmdDispatch(command_buffer, fourierGridSize / 32, fourierGridSize / 32, 1);
		}
		endSingleTimeCommands(command_buffer);
	}

	// Creates a command buffer per image in the swap chain to be able to bind the correct buffer during the draw call
	void createCommandBuffers() {

		commandBuffers.resize(swapChainFramebuffers.size());	// resize the command buffer's array to be equal to the size of the swap chain images

		VkCommandBufferAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;	// Set the type of the struct
		allocInfo.commandPool = commandPool;	// pointer to the command pool from which the command buffers will be allocated from
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;	// primary level allows to be submitted to a queue for execution but it cannot be called from other command's buffers
		allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();	// Is the number of command buffers and will be equal to the number of images in the swap chain

		if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {	// Create the command buffer using the data from the struct
																									// Throw a runtime error if something fails
			throw std::runtime_error("failed to allocate command buffers!");
		}


		VkCommandBufferBeginInfo beginInfo = {};	// specifies the initial state of the command buffer
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;	// Set the type of the struct
		for (size_t i = 0; i < commandBuffers.size(); i++) {	// For every command buffer 

			if (vkBeginCommandBuffer(commandBuffers[i], &beginInfo) != VK_SUCCESS) {	// create the begin command buffer with the data of the struct
				throw std::runtime_error("failed to begin recording command buffer!");
			}
			
			VkBuffer vertexBuffers[] = { vertexBuffer };
			VkDeviceSize offsets[] = { 0 };
			VkBuffer vertexBuffersQuad[] = { vertexBufferQuad };
			VkBuffer vertexBuffersCube[] = { vertexBufferCube };
			
			// Create the Hkt texture
			{
				pushConstantsComputeHktPipeline pcsp = {};
				pcsp.fourierGridSize = fourierGridSize;
				pcsp.spatialDimension = spatialDimension;


				vkCmdPushConstants(
					commandBuffers[i],
					computeHktPipelineLayout,
					VK_SHADER_STAGE_COMPUTE_BIT,
					0,
					sizeof(pushConstantsComputeHktPipeline),
					&pcsp);

				vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_COMPUTE, computeHktPipeline);
				vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_COMPUTE, computeHktPipelineLayout, 0, 1, &descriptorSets[0], 0, nullptr);

				vkCmdDispatch(commandBuffers[i], fourierGridSize / 32, fourierGridSize / 32, 1);
			}


		 if(!isHkt)
			{
				// IFFT algorithm 
				{
					int stageSize = int(log2(fourierGridSize));
					int swap = 0;
					pushConstantsComputeFFTPipeline pcfft = {};

					
					// Horizontal
					for (int k = 0; k < stageSize; k++)
					{
						pcfft.stage = k;
						pcfft.swap = swap;

						vkCmdPushConstants(
							commandBuffers[i],
							computeFFTHorizontalPipelineLayout,
							VK_SHADER_STAGE_COMPUTE_BIT,
							0,
							sizeof(pushConstantsComputeFFTPipeline),
							&pcfft);

						vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_COMPUTE, computeFFTHorizontalPipeline);
						vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_COMPUTE, computeFFTHorizontalPipelineLayout, 0, 1, &descriptorSets[0], 0, nullptr);

						vkCmdDispatch(commandBuffers[i], fourierGridSize / 32, fourierGridSize / 32, 1);

						swap = 1 - swap;
					}
					
					// Vertical 
					for (int k = 0; k < stageSize; k++)
					{
						pcfft.stage = k;
						pcfft.swap = swap;

						vkCmdPushConstants(
							commandBuffers[i],
							computeFFTVerticalPipelineLayout,
							VK_SHADER_STAGE_COMPUTE_BIT,
							0,
							sizeof(pushConstantsComputeFFTPipeline),
							&pcfft);

						vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_COMPUTE, computeFFTVerticalPipeline);
						vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_COMPUTE, computeFFTVerticalPipelineLayout, 0, 1, &descriptorSets[0], 0, nullptr);

						vkCmdDispatch(commandBuffers[i], fourierGridSize / 32, fourierGridSize / 32, 1);

						swap = 1 - swap;
					}
					

					// Inversion
					{
						pushConstantsComputeHeightMapPipeline pchm = {};
						pchm.fourierGridSize = fourierGridSize;
						pchm.swap = swap;

						vkCmdPushConstants(
							commandBuffers[i],
							computeHeightMapPipelineLayout,
							VK_SHADER_STAGE_COMPUTE_BIT,
							0,
							sizeof(pushConstantsComputeHeightMapPipeline),
							&pchm);

						vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_COMPUTE, computeHeightMapPipeline);
						vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_COMPUTE, computeHeightMapPipelineLayout, 0, 1, &descriptorSets[0], 0, nullptr);

						vkCmdDispatch(commandBuffers[i], fourierGridSize / 32, fourierGridSize / 32, 1);
					}

				}

			}
			
			{
				// Drawing will start by performing a render pass
				VkRenderPassBeginInfo renderPassInfo = {};
				renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;	// Set the type of the struct
				renderPassInfo.renderPass = renderPass;	// pointer to the struct that holds the configuration of a render pass
				renderPassInfo.framebuffer = swapChainFramebuffers[i];	// specify which frame buffer will be used for the render pass in our case it will be the attachment of the color buffer
				renderPassInfo.renderArea.offset = { 0, 0 };	// We don't want any offset to the render area because we want it to fit with the one of the attachments
				renderPassInfo.renderArea.extent = swapChainExtent;	// Define the area as the one defined in the swap chain

				std::array<VkClearValue, 2> clearValues = {};
				clearValues[0].color = { 1.0f, 0.8f, 1.0f, 1.0f };	// It will define the color of the background
				clearValues[1].depthStencil = { 1.0f, 0 };

				renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
				renderPassInfo.pClearValues = clearValues.data();

				// records a command in the specified command buffer as the first parameter, the second parameter will be the data of the struct we've just created,
				// The third argument specifies that the render pass commands will only be executed from this command buffer
				vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
				

				// Shadow map display
				if (isDisplayTextures) {
					vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, vertexBuffersQuad, offsets);
					vkCmdBindIndexBuffer(commandBuffers[i], indexBufferQuad, 0, VK_INDEX_TYPE_UINT32);
					
					pushConstantsQuadPipeline[0] = isH0k == true ? 1 : 0;
					pushConstantsQuadPipeline[1] = isH0kMinus == true ? 1 : 0;
					pushConstantsQuadPipeline[2] = isHkt == true ? 1 : 0;
					pushConstantsQuadPipeline[3] = isFFTAux == true ? 1 : 0;
					pushConstantsQuadPipeline[4] = isHorizontalFFT == true ? 1 : 0;
					pushConstantsQuadPipeline[5] = isVerticalFFT == true ? 1 : 0;
					pushConstantsQuadPipeline[6] = isHeightMap == true ? 1 : 0;
					pushConstantsQuadPipeline[7] = isSlopeX == true ? 1 : 0;
					pushConstantsQuadPipeline[8] = isSlopeZ == true ? 1 : 0;
					pushConstantsQuadPipeline[9] = isDispX == true ? 1 : 0;
					pushConstantsQuadPipeline[10] = isDispZ == true ? 1 : 0;


					vkCmdPushConstants(
						commandBuffers[i],
						pipelineLayoutQuad,
						VK_SHADER_STAGE_FRAGMENT_BIT,
						0,
						sizeof(pushConstantsQuadPipeline),
						pushConstantsQuadPipeline.data());

					vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineQuad);
					vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayoutQuad, 0, 1, &descriptorSets[i], 0, NULL);
					vkCmdDrawIndexed(commandBuffers[i], static_cast<uint32_t>(indicesQuad.size()), 1, 0, 0, 0);
				}
				else
				{
					vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, vertexBuffers, offsets);	// Binds the vertex buffer to be used for the draw call
					vkCmdBindIndexBuffer(commandBuffers[i], indexBuffer, 0, VK_INDEX_TYPE_UINT32);

					// send which behaviour will have the shader depending on the gui
					pushConstantsRenderScenePipeline[0] = isWavy == true ? 1 : 0;

					vkCmdPushConstants(
						commandBuffers[i],
						pipelineLayout,
						VK_SHADER_STAGE_VERTEX_BIT,
						0,
						sizeof(pushConstantsRenderScenePipeline),
						pushConstantsRenderScenePipeline.data());
					
					// The first parameter to bind the pipeline is the command buffer that we are configuring, the second one will be if the pipeline is a graphics one or a compute.
					// The third parameter is the data from the struct when we configured the graphics pipeline
					vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
					vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[i], 0, nullptr);

					// Call to finally draw the triangle specifiying first the command buffer,
					// Second is the number of vertex that will be draw
					// Third is for instance rendering, as we are not using it we use 1
					// Fourth defines the lowest vertex that we have to have it as a first vertex
					// Fifth defines the lowest value for instance rendering
					vkCmdDrawIndexed(commandBuffers[i], static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

					
					// CUBEMAP


					vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, vertexBuffersCube, offsets);	// Binds the vertex buffer to be used for the draw call
					vkCmdBindIndexBuffer(commandBuffers[i], indexBufferCube, 0, VK_INDEX_TYPE_UINT32);

					vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineCube);
					vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayoutCube, 0, 1, &descriptorSets[i], 0, nullptr);

					vkCmdDrawIndexed(commandBuffers[i], static_cast<uint32_t>(indicesCube.size()), 1, 0, 0, 0);


				}

				vkCmdEndRenderPass(commandBuffers[i]);
			}
			
			if (vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS) {	// Create the command buffer using the data from the struct 
																		// Throw a runtime error if something fails
				throw std::runtime_error("failed to record command buffer!");
			}
		}
	}

	// Creates all the objects required to create synchronization
	void createSyncObjects() {
		imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);	// resize the number of semaphores to the number of simultaneous frames
		renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);	// resize in the same way for the fences
		imagesInFlight.resize(swapChainImages.size(), VK_NULL_HANDLE);	// resize to have the number of fences equal to the number of images in the swap chain

		// Struct to give the info of the sempahores
		VkSemaphoreCreateInfo semaphoreInfo = {};
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;	// Set the type of the struct

		VkFenceCreateInfo fenceInfo = {};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;	// Set the type of the struct
		fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;	// the fence starts being signaled

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
			if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||		// Create the semaphores using the data from the structs
				vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
				vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {	// Creates the fence using the data from the struct
				throw std::runtime_error("failed to create synchronization objects for a frame!");
			}
		}
	}

	// Updates the information related to the uniform buffer inside of the drawFrame loop
	void updateUniformBuffer(uint32_t currentImage) {
		static auto startTime = std::chrono::high_resolution_clock::now();

		auto currentTime = std::chrono::high_resolution_clock::now();
		float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

		UniformBufferObject ubo = {};	// fill in all the information required to the struct
		glm::vec3 translation(0.0f, -0.5f, -5.0f);
		glm::mat4 model = glm::translate(glm::mat4(1.0f), translation);
		ubo.model = glm::scale(model, { 1, 1, 1 });
		//ubo.model = glm::rotate(ubo.model, glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		//ubo.model = glm::rotate(ubo.model, (time/10) * glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));	// apply some rotation to be different points of view
		
		ubo.modelCube = glm::scale(model, { 1000, 1000, 1000 });
		ubo.view = camera.GetViewMatrix();
		ubo.viewCube = ubo.view;
		ubo.viewCube[3] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
		
		ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float)swapChainExtent.height, 0.1f, 5000.0f);
		ubo.proj[1][1] *= -1;
		ubo.cameraPosition = camera.GetPosition();

		void* data;
		vkMapMemory(device, uniformBuffersMemory[currentImage], 0, sizeof(ubo), 0, &data);
		memcpy(data, &ubo, sizeof(ubo));
		vkUnmapMemory(device, uniformBuffersMemory[currentImage]);
	}

	// Updates the information related to the uniform buffer inside of the drawFrame loop
	void updateLightUniformBuffer(uint32_t currentImage) {
		std::ifstream inFile;  // Input file opening and checking

		inFile.open("oceanMaterial.mtl");
		if (!inFile) {
			throw std::runtime_error("Error opening Model object");
		}

		float specularHighlight;
		glm::vec3 ambientColor;
		glm::vec3 diffuseColor;
		glm::vec3 specularColor;
		glm::vec3 emissiveColor;

		std::string line;
		double x, y, z;

		while (!inFile.eof()) {		// Read the .mtl file with the information of the material
			std::getline(inFile, line);

			if (line[0] == '\t' && line[1] == 'N' && line[2] == 's')
			{
				size_t pos = line.find(' ');
				line.erase(0, pos + 1);
				pos = line.find(' ');
				specularHighlight = std::stof(line.substr(0, pos));
			}
			else if (line[0] == '\t' && line[1] == 'K' && line[2] == 'a')
			{
				size_t pos = line.find(' ');
				line.erase(0, pos + 1);
				pos = line.find(' ');
				x = std::stof(line.substr(0, pos));

				pos = line.find(' ');
				line.erase(0, pos + 1);
				pos = line.find(' ');
				y = std::stof(line.substr(0, pos));

				pos = line.find(' ');
				line.erase(0, pos + 1);
				pos = line.find(' ');
				z = std::stof(line.substr(0, pos));

				ambientColor = { x, y, z };
			}
			else if (line[0] == '\t' && line[1] == 'K' && line[2] == 'd')
			{
				size_t pos = line.find(' ');
				line.erase(0, pos + 1);
				pos = line.find(' ');
				x = std::stof(line.substr(0, pos));

				pos = line.find(' ');
				line.erase(0, pos + 1);
				pos = line.find(' ');
				y = std::stof(line.substr(0, pos));

				pos = line.find(' ');
				line.erase(0, pos + 1);
				pos = line.find(' ');
				z = std::stof(line.substr(0, pos));

				diffuseColor = { x, y, z };
			}
			else if (line[0] == '\t' && line[1] == 'K' && line[2] == 's')
			{
				size_t pos = line.find(' ');
				line.erase(0, pos + 1);
				pos = line.find(' ');
				x = std::stof(line.substr(0, pos));

				pos = line.find(' ');
				line.erase(0, pos + 1);
				pos = line.find(' ');
				y = std::stof(line.substr(0, pos));

				pos = line.find(' ');
				line.erase(0, pos + 1);
				pos = line.find(' ');
				z = std::stof(line.substr(0, pos));

				specularColor = { x, y, z };
			}
			else if (line[0] == '\t' && line[1] == 'K' && line[2] == 'e')
			{
				size_t pos = line.find(' ');
				line.erase(0, pos + 1);
				pos = line.find(' ');
				x = std::stof(line.substr(0, pos));

				pos = line.find(' ');
				line.erase(0, pos + 1);
				pos = line.find(' ');
				y = std::stof(line.substr(0, pos));

				pos = line.find(' ');
				line.erase(0, pos + 1);
				pos = line.find(' ');
				z = std::stof(line.substr(0, pos));

				emissiveColor = { x, y, z };
			}
		}
		inFile.close();
		
		UniformBufferLight ubl = {};	// After reading the file we can fill in all the information that we will use in this UniformBuffer 
		ubl.specularHighlight = specularHighlight;
		ubl.ambientColor = ambientColor;
		ubl.diffuseColor = diffuseColor;
		ubl.specularColor = specularColor;
		ubl.emissiveColor = emissiveColor;
		ubl.lightPosition = lightPosition;

		void* data;
		vkMapMemory(device, lightUniformBuffersMemory[currentImage], 0, sizeof(ubl), 0, &data);
		memcpy(data, &ubl, sizeof(ubl));
		vkUnmapMemory(device, lightUniformBuffersMemory[currentImage]);
	}

	void updateUniformBufferTimeHkt(uint32_t currentImage) {
		
		if(!isTimeStop)
			timeAnimation = float(ImGui::GetTime()*waterSpeed);

		UniformBufferObjectTimeHkt ubthkt = {};
		ubthkt.time = timeAnimation;

		void* data;
		vkMapMemory(device, uniformBuffersTimeHktMemory[currentImage], 0, sizeof(UniformBufferObjectTimeHkt), 0, &data);
		memcpy(data, &ubthkt, sizeof(UniformBufferObjectTimeHkt));
		vkUnmapMemory(device, uniformBuffersTimeHktMemory[currentImage]);
	}

	void createCommandBuffersGUI() 
	{
		
		VkCommandBufferAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;	// Set the type of the struct
		allocInfo.commandPool = commandPoolGUI;	// pointer to the command pool from which the command buffers will be allocated from
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;	// primary level allows to be submitted to a queue for execution but it cannot be called from other command's buffers
		allocInfo.commandBufferCount = (uint32_t)commandBuffersGUI.size();	// Is the number of command buffers and will be equal to the number of images in the swap chain
		
		if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffersGUI.data()) != VK_SUCCESS) {	// Create the command buffer using the data from the struct
																									// Throw a runtime error if something fails
			throw std::runtime_error("failed to allocate command buffers!");
		}

		VkCommandBufferBeginInfo beginInfo = {};	// specifies the initial state of the command buffer
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;	// Set the type of the struct
		for (size_t i = 0; i < commandBuffersGUI.size(); i++)	// For every command buffer 
		{
			if (vkBeginCommandBuffer(commandBuffersGUI[i], &beginInfo) != VK_SUCCESS) {	// create the begin command buffer with the data of the struct
				throw std::runtime_error("failed to begin recording command buffer!");
			}
			
			// Render pass to create the gui with the updated values of that instance
			{
				VkRenderPassBeginInfo renderPassInfo = {};
				renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;	// Set the type of the struct
				renderPassInfo.renderPass = renderPassGUI;	// pointer to the struct that holds the configuration of a render pass
				renderPassInfo.framebuffer = swapChainFramebuffersGUI[i];	// specify which frame buffer will be used for the render pass in our case it will be the attachment of the color buffer
				renderPassInfo.renderArea.offset = { 0, 0 };	// We don't want any offset to the render area because we want it to fit with the one of the attachments
				renderPassInfo.renderArea.extent = swapChainExtent;	// Define the area as the one defined in the swap chain

				std::array<VkClearValue, 1> clearValues = {};
				clearValues[0].color = { 1.0f, 0.8f, 1.0f, 1.0f };

				renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
				renderPassInfo.pClearValues = clearValues.data();

				vkCmdBeginRenderPass(commandBuffersGUI[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

				// call to the gui to draw its values
				ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffersGUI[i]);

				vkCmdEndRenderPass(commandBuffersGUI[i]);
			}

			if (vkEndCommandBuffer(commandBuffersGUI[i]) != VK_SUCCESS) {	// Create the command buffer using the data from the struct 
																		// Throw a runtime error if something fails
				throw std::runtime_error("failed to record command buffer!");
			}
		}
	}

	// create a separate command pool that resets its values on drawing
	void createCommandPoolGUI(VkCommandPoolCreateFlags flags) {
		VkCommandPoolCreateInfo commandPoolCreateInfo = {};
		commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		commandPoolCreateInfo.queueFamilyIndex = findQueueFamilies(physicalDevice).graphicsFamily.value();
		commandPoolCreateInfo.flags = flags;

		if (commandPoolGUI)
		{
			vkDeviceWaitIdle(device);
			vkDestroyCommandPool(device, commandPoolGUI, nullptr);
		}

		if (vkCreateCommandPool(device, &commandPoolCreateInfo, nullptr, &commandPoolGUI) != VK_SUCCESS) {
			throw std::runtime_error("Could not create graphics command pool");
		}
	}


	// First it will acquire an image from the swap chain,
	// Execute the command buffer using that image as attachment
	// Return the image to the swap chain for presentation 
	// All of this will be done asynchronously, but we will need semaphores and fences to control them
	void drawFrame() {
		vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);	// Waits until either all or any of the fences to be signaled

		uint32_t imageIndex;
		// The first two parameters are the logical device and the swap chain from where we want to acquire images from
		// The third parameter is the time out that it has to get an image from the swap chain, 
		// The fourth parameter is the semaphore that should be signaled when the image is acquired
		// The last parameter is an output variable that will have the index of the retrieved image from the swap chain
		VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

		if (result == VK_ERROR_OUT_OF_DATE_KHR) {	// If the swap chain has become incompatible with the window surface we recreate the swap chain
			recreateSwapChain();
			return;
		}
		else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {	// If there is any error with the swap chain acquiring an image throw a runtime error
			throw std::runtime_error("failed to acquire swap chain image!");
		}

		// Update all the information of the uniforms
		updateUniformBuffer(imageIndex);
		updateLightUniformBuffer(imageIndex);
		updateUniformBufferTimeHkt(imageIndex);


		if (imagesInFlight[imageIndex] != VK_NULL_HANDLE) {	// We check if a previous frame is using this image 
			vkWaitForFences(device, 1, &imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);	// if it is we wait untill it finishes
		}
		imagesInFlight[imageIndex] = inFlightFences[currentFrame];	// Set the image to be in use in this frame

		createCommandPoolGUI(VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
		commandBuffersGUI.resize(swapChainImageViews.size());
		createCommandBuffersGUI();

		// create an array of command buffers to have the gui with the scene. Seems that the priority is from right to left
		std::array<VkCommandBuffer, 2> submitCommandBuffers =
		{ commandBuffers[imageIndex], commandBuffersGUI[imageIndex] };

		// Handles queue submissions and synchronization
		VkSubmitInfo submitInfo = {};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;	// Set the type of the struct

		VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };	// Array containing one of the semaphores
		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };	// Specify the stage of the pipeline after blending 
		submitInfo.waitSemaphoreCount = 1;	// It is the number of semaphores that will wait
		submitInfo.pWaitSemaphores = waitSemaphores;	// pointer to the array of semaphores that will wait
		submitInfo.pWaitDstStageMask = waitStages;	// Specify in which stage will they wait

		submitInfo.commandBufferCount = static_cast<uint32_t>(submitCommandBuffers.size()); // is the number of command buffers to execute in the batch
		submitInfo.pCommandBuffers = submitCommandBuffers.data();	// pointer to the command buffer to use

		VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] };	// Specify an array containing one semaphore for the render finish
		submitInfo.signalSemaphoreCount = 1;	// is the number of semaphores that should be signaled after the execution of the command
		submitInfo.pSignalSemaphores = signalSemaphores;	// pointer to the array of semaphores that need to be signaled

		vkResetFences(device, 1, &inFlightFences[currentFrame]);	// Restore the fences to be unsignaled

		if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {	 // Submit the command buffer to the graphics queue
			throw std::runtime_error("failed to submit draw command buffer!");							 // Throw a runtime error if something fails
		}

		// Struct to configure the presentation of the frame back to the swap chain to be draw
		VkPresentInfoKHR presentInfo = {};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;	// Set the type of the struct

		presentInfo.waitSemaphoreCount = 1;	// Specify the number of semaphores that it should wait before starting
		presentInfo.pWaitSemaphores = signalSemaphores;	// Specify which semaphores it should wait before starting

		VkSwapchainKHR swapChains[] = { swapChain };	// Array containing the swap chain
		presentInfo.swapchainCount = 1;	// Specifies the number of swap chains to present images to 
		presentInfo.pSwapchains = swapChains; // Specifies the swap chains to present images to

		presentInfo.pImageIndices = &imageIndex;	// pointer to the image that will be present it to the swap chain

		result = vkQueuePresentKHR(presentQueue, &presentInfo);	// Submit the request to present an image to the swap chain using the data from the struct

		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {	// If there is an error with the swap chain becomming incompatible with the window
																										// surface or the result is suboptimal or the framebuffer has been resized
																										// recreate the swap chain
			framebufferResized = false;
			recreateSwapChain();
		}
		else if (result != VK_SUCCESS) {	// If there is any error throw a runtime error 
			throw std::runtime_error("failed to present swap chain image!");
		}
		currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;	// update the current frame
	}

	// Wrapper of the bytecode to create a VkShaderModule object
	VkShaderModule createShaderModule(const std::vector<char>& code) {
		VkShaderModuleCreateInfo createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;	// set the type of the struct
		createInfo.codeSize = code.size();	//	size in bytes of the bytecode passed as argument
		createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());	// pointer to the code that will use the VkShaderModule object

		VkShaderModule shaderModule;
		if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {	// Create the shader module with the data in the stuct create info. If there is an error
																								// throw a runtime error
			throw std::runtime_error("failed to create shader module!");
		}

		return shaderModule;
	}

	//Choose which should be the color depth based on the format
	VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
		for (const auto& availableFormat : availableFormats) {	// search for the desired format and colorSpace
			if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
				// if the format is BGRA and the colorspace supports SRGB return them
				return availableFormat;
			}
		}

		return availableFormats[0];	// return the first format in the array. As we don't have our desired we can pick the first one for example as there always will be one
	}

	// Indicates in which way will the images be shown to the screen
	VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
		for (const auto& availablePresentMode : availablePresentModes) { // search for the desired present mode
			if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {	// the mode that we picked: represents the swap chain as a queue where the images are taken from the front 
																		// of the queue when the display is refreshed and if the queue is full the new images will replace
																		// the old ones allowing triple buffering
				return availablePresentMode;
			}
		}

		return VK_PRESENT_MODE_FIFO_KHR;	// if the MailBox present mode is not available it will use the FIFO where when an image is ready 
											// it will be sent to be displayed causing tearing 
	}

	// Choose which resolution it will use for the images in the swap chain 
	VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
		if (capabilities.currentExtent.width != UINT32_MAX) {	// if a width of the window has been selected (different from MAX) it will use that one that fits the current window
			return capabilities.currentExtent;
		}
		else {	// if it is not the case we will have to guess which is the best resolution
			int width, height;
			glfwGetFramebufferSize(window, &width, &height);	// Get the size of the window with the help of GLFW

			VkExtent2D actualExtent = {	// fit the information in a struct
				static_cast<uint32_t>(width),
				static_cast<uint32_t>(height)
			};

			actualExtent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actualExtent.width));	// get the width between the limits and the actual extent
			actualExtent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actualExtent.height)); // get the height between the limits and the actual extent

			return actualExtent;	// return the final selected resolution
		}
	}

	// Query the information need it for filling in the details for the swap chain
	SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
		SwapChainSupportDetails details;

		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);	// retrieves the basic surface capabilities

		uint32_t formatCount;
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);	// get the number of suface formats

		if (formatCount != 0) {	// if there are any format
			details.formats.resize(formatCount);	// resize an actual array of the struct to have the size of the number of surface formats
			vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());	// set the data of each format inside of the array
		}

		uint32_t presentModeCount;
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);	// get the number of present modes

		if (presentModeCount != 0) {	// if there are any present mode
			details.presentModes.resize(presentModeCount);	// resize an actual array of the struct to have the size of the number of present modes
			vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());	// set the data of each present mode inside of the array
		}

		return details;	// return the struct with all the data related to the details of the swap chain
	}

	// Checks if the graphics card "device" is suitable for the purpose of the program that will be built
	bool isDeviceSuitable(VkPhysicalDevice device) {
		QueueFamilyIndices indices = findQueueFamilies(device); // ask if the device is able to proccess the commands that we want

		bool extensionsSupported = checkDeviceExtensionSupport(device);	// checks if the device support all the required extensions

		bool swapChainAdequate = false;
		if (extensionsSupported) {	// if it support all required extensions
			SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);	// retrieve the details of the swap chain
			swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();	// Set the swap chain to be adequate 
																												// if there is at least a format and a present mode
		}
		VkPhysicalDeviceFeatures supportedFeatures;
		vkGetPhysicalDeviceFeatures(device, &supportedFeatures);

		return indices.isComplete() && extensionsSupported && swapChainAdequate && supportedFeatures.samplerAnisotropy;	// if everything is supported and adequate and we have the indices of the queues return true
	}

	// check if the device is able to provide the different extensions required 
	bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
		uint32_t extensionCount;
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);	// get the number of extensions in the system

		std::vector<VkExtensionProperties> availableExtensions(extensionCount);	// create an array to store the data of every extension
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());	// fill in the data of every extension in the system into the array 

		std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());	// create a set of strings with the extensions that the device supports

		for (const auto& extension : availableExtensions) {	// for each extension of the system that the device supports
			requiredExtensions.erase(extension.extensionName);	// erase the extensions from the list of required extensions
		}

		return requiredExtensions.empty();	// if the list of requiered extensions is empty it means that all of them are supported
	}

	QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
		QueueFamilyIndices indices;	// It will hold the queues indexes 

		uint32_t queueFamilyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);	// look for the number of queues available to the system

		std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);	// array that holds the details of every queue
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data()); // fill in the array with the values of every queue

		int i = 0;
		for (const auto& queueFamily : queueFamilies) {	// check for every queue available to the system
			if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {	// if there is the one related to the graphics commands
				indices.graphicsFamily = i;	// assing the index of the loop to the graphics queue
			}

			VkBool32 presentSupport = false;
			vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);	// search for a queue able to present to our window surface

			if (presentSupport) {	// if the queue for presenting was found
				indices.presentFamily = i;	// set the index to the one in the loop
			}

			if (indices.isComplete()) {	// Check if we have already find all the queues that we wanted so we don't need to keep looping.
				break;
			}

			i++;
		}

		return indices;	// return the updated values of the queues
	}

	std::vector<const char*> getRequiredExtensions() {
		uint32_t glfwExtensionCount = 0;
		const char** glfwExtensions;
		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);	// GLFW get for us the extensions that are need it

		std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);	// vector that will hold the extensions that our application is going to have

		if (enableValidationLayers) {	// if validation layers are activated
			extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);	// add this extension to the extension list
		}

		return extensions;	// return all the extensions 
	}

	// Checks if all requested validations layer are available to the program
	bool checkValidationLayerSupport() {
		uint32_t layerCount;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);	// We get the number of layer properties available to the system

		std::vector<VkLayerProperties> availableLayers(layerCount);	// create a vector with the size of the number of layer properties available, in order
																	// to hold the data of each of them
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());	// fill in the data of each available layer properties in the vector.

		// Checks if every desired validation layer to the system is in the list of the available layers to the system
		for (const char* layerName : validationLayers) {
			bool layerFound = false;

			for (const auto& layerProperties : availableLayers) {
				if (strcmp(layerName, layerProperties.layerName) == 0) {
					layerFound = true;
					break;
				}
			}

			if (!layerFound) {	// if any of the layers fail it returns false meaning that not all desired validation layers are available.
				return false;
			}
		}

		return true;	// otherwise everything is as we would like and it return true (success)
	}

	// It will read the binary files for the shaders from end to beginning
	static std::vector<char> readFile(const std::string& filename) {
		std::ifstream file(filename, std::ios::ate | std::ios::binary);	// open the file from the end (ate) and in binary format to avoid text transformations

		if (!file.is_open()) {	// if the file is not correctly opened throw a runtime error
			throw std::runtime_error("failed to open file!");
		}

		size_t fileSize = (size_t)file.tellg();	// As we started the file from the end the pointer of the file tell us its size
		std::vector<char> buffer(fileSize);	// Create the buffer with the size of the input file 

		file.seekg(0);	// Go to the beginning of the file
		file.read(buffer.data(), fileSize);	// get the data from the file that was opened and throw it into the buffer

		file.close();	// close the file

		return buffer;
	}

	// the VKAPI_ATTR, VKAPI_CALL are need it for Vulkan to correctly call it
	// the first parameter tells the severity of the message. ie. error, warning, diagnostic...
	// the type can be: unrelated to performance nor specificatione erro, violates specification, non optimal performance
	// pCallbackData contains details of the message itself. If it is a string, array of Vulkan objects or number of objects in an array
	// pUserData a pointer to allow passing data to it
	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
		std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;	// output the message data

		return VK_FALSE;	// this return tells if the program should be aborted
	}
};

int main() {
	HelloTriangleApplication app;

	try {

		app.run();	// execute the initialization, the loop, and the cleanup
	}
	catch (const std::exception& e) {	// Catch any type of error
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}