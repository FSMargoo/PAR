#include <array>
#include <cmath>
#include <ctime>
#include <format>
#include <vector>
#include <windows.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <backends/imgui_impl_opengl3.h>
#include <backends/imgui_impl_glfw.h>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/gtc/constants.hpp>

#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <ctime>

// Forward declarations
class ShaderProgram;
class ComputeShader;
class SSBO;

GLFWwindow *InitializeWindow();

std::string LoadShaderSource(const char *filename);

void DrawFPSChart(const std::vector<float> &fpsHistory);

/**
 * @struct BoxData
 * @brief Data structure representing a box/rectangle shape with position, color, and light properties
 */
struct BoxData {
    glm::vec2 leftTop;     ///< Top-left corner coordinates (normalized)
    glm::vec2 rightBottom; ///< Bottom-right corner coordinates (normalized)
    glm::vec3 color;       ///< RGB color of the box
    uint32_t  isLight;     ///< Flag indicating if the box is a light source
};

/**
 * @struct TriangleData
 * @brief Data structure representing a triangle shape with vertices, color, and light properties
 */
struct TriangleData {
    glm::vec2 p1;      ///< First vertex coordinates (normalized)
    glm::vec2 p2;      ///< Second vertex coordinates (normalized)
    glm::vec2 p3;      ///< Third vertex coordinates (normalized)
    glm::vec3 color;   ///< RGB color of the triangle
    uint32_t  isLight; ///< Flag indicating if the triangle is a light source
};

/**
 * @struct CircleData
 * @brief Data structure representing a circle shape with center, radius, color, and light properties
 */
struct CircleData {
    glm::vec2 center;  ///< Center coordinates (normalized)
    glm::vec2 radius;  ///< Radius values (x and y axes)
    glm::vec3 color;   ///< RGB color of the circle
    uint32_t  isLight; ///< Flag indicating if the circle is a light source
};

/**
 * @class SSBO
 * @brief Encapsulates OpenGL Shader Storage Buffer Object functionality
 */
class SSBO {
public:
    /**
     * @brief Constructs an SSBO with specified binding point and initial data
     * @param bindingPoint The binding point index for the SSBO
     * @param data Pointer to the initial data
     * @param size Size of the data in bytes
     * @param usage Usage pattern (default: GL_DYNAMIC_DRAW)
     */
    SSBO(GLuint bindingPoint, const void *data, size_t size, GLenum usage = GL_DYNAMIC_DRAW)
        : bindingPoint(bindingPoint) {
        glGenBuffers(1, &ssboId);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboId);
        glBufferData(GL_SHADER_STORAGE_BUFFER, size, data, usage);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bindingPoint, ssboId);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }

    /**
     * @brief Destructor - releases OpenGL resources
     */
    ~SSBO() {
        if (ssboId != 0) {
            glDeleteBuffers(1, &ssboId);
        }
    }

    /**
     * @brief Updates the data in the SSBO
     * @param data Pointer to the new data
     * @param size Size of the data in bytes
     * @param offset Offset in the buffer where to start updating
     */
    void UpdateData(const void *data, size_t size, size_t offset = 0) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboId);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, offset, size, data);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }

    /**
     * @brief Binds the SSBO to its binding point
     */
    void Bind() const {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bindingPoint, ssboId);
    }

    /**
     * @brief Gets the SSBO ID
     * @return OpenGL buffer identifier
     */
    GLuint GetId() const {
        return ssboId;
    }

private:
    GLuint ssboId;       ///< OpenGL buffer identifier
    GLuint bindingPoint; ///< Binding point index
};

/**
 * @class ShaderProgram
 * @brief Encapsulates OpenGL shader program functionality
 */
class ShaderProgram {
public:
    /**
     * @brief Constructs a shader program from vertex and fragment shader files
     * @param vertexShaderPath Path to the vertex shader file
     * @param fragmentShaderPath Path to the fragment shader file
     */
    ShaderProgram(const std::string &vertexShaderPath, const std::string &fragmentShaderPath) {
        // Load and compile shaders
        std::string vertexSource   = LoadShaderSource(vertexShaderPath.c_str());
        std::string fragmentSource = LoadShaderSource(fragmentShaderPath.c_str());

        GLuint vertexShader   = CompileShader(GL_VERTEX_SHADER, vertexSource.c_str());
        GLuint fragmentShader = CompileShader(GL_FRAGMENT_SHADER, fragmentSource.c_str());

        // Create and link program
        programId = glCreateProgram();
        glAttachShader(programId, vertexShader);
        glAttachShader(programId, fragmentShader);
        glLinkProgram(programId);

        // Check for linking errors
        int  success;
        char infoLog[512];
        glGetProgramiv(programId, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(programId, 512, nullptr, infoLog);
            std::cerr << "Shader program linking error:\n" << infoLog << std::endl;
        }

        // Clean up shaders
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
    }

    /**
     * @brief Uses the shader program
     */
    void Use() const {
        glUseProgram(programId);
    }

    /**
     * @brief Gets the program ID
     * @return OpenGL program identifier
     */
    GLuint GetId() const {
        return programId;
    }

    /**
     * @brief Sets a uniform integer value
     * @param name Name of the uniform variable
     * @param value Integer value to set
     */
    void SetInt(const std::string &name, int value) const {
        glUniform1i(glGetUniformLocation(programId, name.c_str()), value);
    }

    /**
     * @brief Sets a uniform float value
     * @param name Name of the uniform variable
     * @param value Float value to set
     */
    void SetFloat(const std::string &name, float value) const {
        glUniform1f(glGetUniformLocation(programId, name.c_str()), value);
    }

private:
    GLuint programId; ///< OpenGL program identifier

    /**
     * @brief Compiles a shader from source code
     * @param type Type of shader (GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, etc.)
     * @param source Source code of the shader
     * @return Compiled shader identifier
     */
    GLuint CompileShader(GLenum type, const char *source) {
        GLuint shader = glCreateShader(type);
        glShaderSource(shader, 1, &source, nullptr);
        glCompileShader(shader);

        // Check for compilation errors
        int  success;
        char infoLog[512];
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(shader, 512, nullptr, infoLog);
            std::cerr << "Shader compilation error:\n" << infoLog << std::endl;
        }

        return shader;
    }
};

/**
 * @class ComputeShader
 * @brief Encapsulates OpenGL compute shader functionality
 */
class ComputeShader {
public:
    /**
     * @brief Constructs a compute shader from a file
     * @param computeShaderPath Path to the compute shader file
     */
    ComputeShader(const std::string &computeShaderPath) {
        // Load and compile shader
        std::string computeSource = LoadShaderSource(computeShaderPath.c_str());
        GLuint      computeShader = CompileShader(GL_COMPUTE_SHADER, computeSource.c_str());

        // Create and link program
        programId = glCreateProgram();
        glAttachShader(programId, computeShader);
        glLinkProgram(programId);

        // Check for linking errors
        int  success;
        char infoLog[512];
        glGetProgramiv(programId, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(programId, 512, nullptr, infoLog);
            std::cerr << "Compute program linking error:\n" << infoLog << std::endl;
        }

        // Clean up shader
        glDeleteShader(computeShader);
    }

    /**
     * @brief Uses the compute shader program
     */
    void Use() const {
        glUseProgram(programId);
    }

    /**
     * @brief Gets the program ID
     * @return OpenGL program identifier
     */
    GLuint GetId() const {
        return programId;
    }

private:
    GLuint programId; ///< OpenGL program identifier

    /**
     * @brief Compiles a shader from source code
     * @param type Type of shader (GL_COMPUTE_SHADER)
     * @param source Source code of the shader
     * @return Compiled shader identifier
     */
    GLuint CompileShader(GLenum type, const char *source) {
        GLuint shader = glCreateShader(type);
        glShaderSource(shader, 1, &source, nullptr);
        glCompileShader(shader);

        // Check for compilation errors
        int  success;
        char infoLog[512];
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(shader, 512, nullptr, infoLog);
            std::cerr << "Shader compilation error:\n" << infoLog << std::endl;
        }

        return shader;
    }
};

typedef struct tagRECTF {
	double left;
	double top;
	double right;
	double bottom;
} RECTF, *PRECTF;

// 2D Vector
class Vec final {
public:
	Vec() : x({0, 0}) {
	}
	Vec(std::array<double, 2> Init) : x(Init) {
	}

public:
	auto Length() noexcept -> double {
		return sqrt(pow(x[0], 2) + pow(x[1], 2));
	}
	auto Normalize() noexcept -> Vec {
		return (*this) / Length();
	}
	auto DotProduct(const Vec &Vector) noexcept -> double {
		return Vector.x[0] * x[0] + Vector.x[1] * x[1];
	}

public:
	auto operator*=(const Vec &Value) -> Vec & {
		for (auto position = size_t(0); position < 2; ++position) {
			x[position] *= Value.x[position];
		}

		return (*this);
	}
	auto operator/=(const Vec &Value) -> Vec & {
		for (auto position = size_t(0); position < 2; ++position) {
			x[position] /= Value.x[position];
		}

		return (*this);
	}
	auto operator+=(const Vec &Value) -> Vec & {
		for (auto position = size_t(0); position < 2; ++position) {
			x[position] += Value.x[position];
		}

		return (*this);
	}
	auto operator-=(const Vec &Value) -> Vec & {
		for (auto position = size_t(0); position < 2; ++position) {
			x[position] -= Value.x[position];
		}

		return (*this);
	}
	auto operator*=(const double &Value) -> Vec & {
		for (auto position = size_t(0); position < 2; ++position) {
			x[position] * Value;
		}

		return (*this);
	}
	auto operator/=(const double &Value) -> Vec & {
		for (auto position = size_t(0); position < 2; ++position) {
			x[position] / Value;
		}

		return (*this);
	}
	auto operator+=(const double &Value) -> Vec & {
		for (auto position = size_t(0); position < 2; ++position) {
			x[position] + Value;
		}

		return (*this);
	}
	auto operator-=(const double &Value) -> Vec & {
		for (auto position = size_t(0); position < 2; ++position) {
			x[position] - Value;
		}

		return (*this);
	}

public:
	friend auto operator/(const Vec &Left, const Vec &Right) -> Vec;
	friend auto operator/(const Vec &Left, const double &Right) -> Vec;
	friend auto operator*(const Vec &Left, const Vec &Right) -> Vec;
	friend auto operator*(const Vec &Left, const double &Right) -> Vec;
	friend auto operator+(const Vec &Left, const Vec &Right) -> Vec;
	friend auto operator+(const Vec &Left, const double &Right) -> Vec;
	friend auto operator-(const Vec &Left, const Vec &Right) -> Vec;
	friend auto operator-(const Vec &Left, const double &Right) -> Vec;

public:
	std::array<double, 2> x;
};

auto operator/(const Vec &Left, const Vec &Right) -> Vec {
	Vec result;
	for (auto position = size_t(0); position < 2; ++position) {
		result.x[position] = Left.x[position] / Right.x[position];
	}

	return result;
}
auto operator/(const Vec &Left, const double &Right) -> Vec {
	Vec result;
	for (auto position = size_t(0); position < 2; ++position) {
		result.x[position] = Left.x[position] / Right;
	}

	return result;
}
auto operator*(const Vec &Left, const Vec &Right) -> Vec {
	Vec result;
	for (auto position = size_t(0); position < 2; ++position) {
		result.x[position] = Left.x[position] * Right.x[position];
	}

	return result;
}
auto operator*(const Vec &Left, const double &Right) -> Vec {
	Vec result;
	for (auto position = size_t(0); position < 2; ++position) {
		result.x[position] = Left.x[position] * Right;
	}

	return result;
}
auto operator+(const Vec &Left, const Vec &Right) -> Vec {
	Vec result;
	for (auto position = size_t(0); position < 2; ++position) {
		result.x[position] = Left.x[position] + Right.x[position];
	}

	return result;
}
auto operator+(const Vec &Left, const double &Right) -> Vec {
	Vec result;
	for (auto position = size_t(0); position < 2; ++position) {
		result.x[position] = Left.x[position] + Right;
	}

	return result;
}
auto operator-(const Vec &Left, const Vec &Right) -> Vec {
	Vec result;
	for (auto position = size_t(0); position < 2; ++position) {
		result.x[position] = Left.x[position] - Right.x[position];
	}

	return result;
}
auto operator-(const Vec &Left, const double &Right) -> Vec {
	Vec result;
	for (auto position = size_t(0); position < 2; ++position) {
		result.x[position] = Left.x[position] - Right;
	}

	return result;
}

auto operator/(const double &Right, const Vec &Left) -> Vec {
	Vec result;
	for (auto position = size_t(0); position < 2; ++position) {
		result.x[position] = Right / Left.x[position];
	}

	return result;
}
auto operator*(const double &Right, const Vec &Left) -> Vec {
	Vec result;
	for (auto position = size_t(0); position < 2; ++position) {
		result.x[position] = Left.x[position] * Right;
	}

	return result;
}
auto operator+(const double &Right, const Vec &Left) -> Vec {
	Vec result;
	for (auto position = size_t(0); position < 2; ++position) {
		result.x[position] = Left.x[position] + Right;
	}

	return result;
}
auto operator-(const double &Right, const Vec &Left) -> Vec {
	Vec result;
	for (auto position = size_t(0); position < 2; ++position) {
		result.x[position] = Right - Left.x[position];
	}

	return result;
}

class Sprite {
public:
	Sprite() = default;

public:
	// Use SDF for collision judgement
	virtual auto SDF(const Vec &Point) -> double	  = 0;
	virtual auto GradientSDF(const Vec &Point) -> Vec = 0;
	virtual auto DealSDF(const double &SDF) -> double {
		return SDF;
	}
	virtual auto Particle() -> Vec						   = 0;
	virtual auto RelativeMove(const Vec &Position) -> void = 0;

public:
	auto Move(const double &X, const double &Y) -> void {
		auto width	= boundingBox.right - boundingBox.left;
		auto height = boundingBox.bottom - boundingBox.top;

		boundingBox = {X, Y, X + width, Y + height};
	}

public:
	bool  lock = true;
	RECTF boundingBox{};
	Vec	  velocity;
};
class RoundSprite : public Sprite {
public:
	explicit RoundSprite(const double &Radius) : radius(Radius), Sprite() {
		boundingBox = {0, 0, Radius * 2, Radius * 2};
	}

public:
	auto SDF(const Vec &Point) -> double override {
		Vec centre({boundingBox.left + radius, boundingBox.top + radius});

		return (Point - centre).Length() - radius;
	}
	auto GradientSDF(const Vec &Point) -> Vec override {
		auto base = SDF(Point) + radius;
		Vec	 centre({boundingBox.left + radius, boundingBox.top + radius});
		return Vec({(Point.x[0] - centre.x[0]) / base, (Point.x[1] - centre.x[1]) / base});
	}
	auto DealSDF(const double &SDF) -> double override {
		return SDF - radius;
	}
	auto Particle() -> Vec override {
		return Vec({boundingBox.left + radius, boundingBox.top + radius});
	}
	auto RelativeMove(const Vec &Position) -> void override {
		Move(Position.x[0] - radius, Position.x[1] - radius);
	}

public:
	double radius;
};
class LineSprite : public Sprite {
public:
	explicit LineSprite(const Vec &Point1, const Vec &Point2) : point1(Point1), point2(Point2) {
		boundingBox = {Point1.x[0], Point1.x[1], Point2.x[0], Point2.x[1]};
	}

public:
	auto SDF(const Vec &Point) -> double override {
		Vec	   ap = Point - point1;
		Vec	   ab = point2 - point1;
		double h  = ap.DotProduct(ab) / ab.DotProduct(ab);
		h		  = h >= 1.f ? 1.f : h;
		h		  = h <= 0.f ? 0.f : h;

		return (ap - h * ab).Length();
	}
	auto GradientSDF(const Vec &Point) -> Vec override {
		auto origin = SDF(Vec({Point.x[0], Point.x[1]}));
		return Vec({SDF(Vec({Point.x[0] + 0.0000000001, Point.x[1]})) - origin,
					SDF(Vec({Point.x[0], Point.x[1] + 0.0000000001})) - origin});
	}
	auto Particle() -> Vec override {
		return point1 + (point2 - point1) / 2;
	}
	auto RelativeMove(const Vec &Position) -> void override {
		auto width	= boundingBox.right - boundingBox.left;
		auto height = boundingBox.bottom - boundingBox.top;
		point1		= Position;
		point2		= point1 + Vec({width, height});
		Move(Position.x[0], Position.x[1]);
	}

public:
	Vec point1;
	Vec point2;
};

class SpriteManager {
public:
	SpriteManager() = default;

public:
	auto UpdateSprite() -> void {
		for (auto &sprite : spriteList) {
			if (!sprite->lock) {
				for (auto &other : spriteList) {
					if (&other == &sprite) {
						continue;
					}
					auto spritePoint = sprite->Particle();
					auto sdf		 = sprite->DealSDF(other->SDF(spritePoint));
					if (sdf <= 0.001) {
						// Normal vector
						auto normal		 = other->GradientSDF(spritePoint).Normalize();
						auto newPosition = spritePoint + normal * abs(sdf);
						// Fix the position
						sprite->RelativeMove(newPosition);

						// Fix the speed
						if (sprite->velocity.DotProduct(normal) < 0) {
							Vec NVelocity	 = sprite->velocity.DotProduct(normal) * normal;
							Vec TVelocity	 = sprite->velocity - NVelocity;
							Vec newNVelocity = (0.f - NVelocity);
							Vec newTVelocity = std::max(1.f - (NVelocity.Length() / TVelocity.Length()), 1.0) * TVelocity;
							sprite->velocity = newNVelocity + newTVelocity;
						}
					}
				}
			}
		}
		for (auto &sprite : spriteList) {
			auto newPosition =
				Vec({static_cast<double>(sprite->boundingBox.left), static_cast<double>(sprite->boundingBox.top)}) +
				sprite->velocity * timingTick;
			sprite->Move(newPosition.x[0], newPosition.x[1]);
		}
	}

public:
	double				  timingTick = 1.f;
	std::vector<Sprite *> spriteList;
};

int main() {
	SpriteManager manager;

	auto roundSpriteSet = new RoundSprite[7]{ RoundSprite(60.f), RoundSprite(60.f), RoundSprite(70.f), RoundSprite(50.f), RoundSprite(40.f), RoundSprite(40.f), RoundSprite(40.f) };
	auto border1        = new LineSprite(Vec({0, 0}), Vec({static_cast<double>(1440), 0}));
	auto border2        = new LineSprite(Vec({0, 0}), Vec({0, static_cast<double>(1440)}));
	auto border3        = new LineSprite(Vec({static_cast<double>(1440), 0}),
												Vec({static_cast<double>(1440), static_cast<double>(1440)}));
	auto		  border4	   = new LineSprite(Vec({0, static_cast<double>(1240)}),
												Vec({static_cast<double>(1440), static_cast<double>(1240)}));
	manager.spriteList.push_back(&roundSpriteSet[0]);
	manager.spriteList.push_back(border1);
	manager.spriteList.push_back(border2);
	manager.spriteList.push_back(border3);
	manager.spriteList.push_back(border4);
	manager.spriteList.push_back(&roundSpriteSet[1]);
	manager.spriteList.push_back(&roundSpriteSet[2]);
	manager.spriteList.push_back(&roundSpriteSet[3]);
	manager.spriteList.push_back(&roundSpriteSet[4]);
	manager.spriteList.push_back(&roundSpriteSet[5]);
	manager.spriteList.push_back(&roundSpriteSet[6]);

	roundSpriteSet[1].Move(180.f * 3.f, 80.f * 10.f);
	roundSpriteSet[2].Move(380.f * 2.f, 180.f * 2.f);
	roundSpriteSet[3].Move(300.f * 2.f, 120.f * 2.f);
	roundSpriteSet[6].Move(400.f * 2.f, 320.f * 2.f);
	roundSpriteSet[0].Move(20.f * 2.5f, 20.f * 4.6f);
	roundSpriteSet[0].velocity.x = {1.5f * 10.f, 0.6f * 10.f};
	roundSpriteSet[3].velocity.x = {-1.7f * 10.f, 1.3f * 10.f};
	roundSpriteSet[4].velocity.x = {-1.5f * 10.f, -2.3f * 10.f};
	roundSpriteSet[5].velocity.x = {3.f * 10.f, -2.3f * 10.f};
	roundSpriteSet[6].velocity.x = {3.f * 10.f, -3.0f * 10.f};
	roundSpriteSet[3].lock		 = false;
	roundSpriteSet[0].lock		 = false;
	roundSpriteSet[4].lock		 = false;
	roundSpriteSet[5].lock		 = false;
	roundSpriteSet[6].lock		 = false;

	// Initialize GLFW window and OpenGL context
    GLFWwindow *window = InitializeWindow();
    if (window == nullptr) {
        return -1;
    }

    // Create shader programs
    ShaderProgram sceneProgram("../shader/vertex.glsl", "../shader/frag.glsl");

    // Define quad vertices and indices for full-screen rendering
    float quadVertices[] = {
        -1.0f, -1.0f, 0.0f, 0.0f, // Bottom-left
        1.0f, -1.0f, 1.0f, 0.0f,  // Bottom-right
        1.0f, 1.0f, 1.0f, 1.0f,   // Top-right
        -1.0f, 1.0f, 0.0f, 1.0f   // Top-left
    };

    unsigned int quadIndices[] = {
        0, 1, 2, // First triangle
        2, 3, 0  // Second triangle
    };

    // Create and configure VAO, VBO, and EBO
    unsigned int VAO, VBO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(quadIndices), quadIndices, GL_STATIC_DRAW);

    // Set vertex attribute pointers
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *) 0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *) (2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
    // Define scene geometry

    std::vector<BoxData> boxes = {
    };

    std::vector<TriangleData> triangles;

    std::vector<CircleData> circles = {
    };

	srand(time(nullptr));
	for (int count = 0; count < 7; ++count) {
		circles.push_back(CircleData{ .center = { (roundSpriteSet[count].boundingBox.left + roundSpriteSet[count].boundingBox.right) / 2.0  / 1440.f,
				(roundSpriteSet[count].boundingBox.top + roundSpriteSet[count].boundingBox.bottom) / 2.0  / 1440.f
			},
			.radius = {roundSpriteSet[count].radius  / 1440.f, roundSpriteSet[count].radius  / 1440.f},
			.color = glm::vec3((rand() % 255) / 255.f, (rand() % 255) / 255.f, (rand() % 255) / 255.f),
			.isLight = !roundSpriteSet[count].lock
		});
	}

    // Create SSBOs for scene geometry using SSBO class
    SSBO boxSSBO(0, boxes.data(), boxes.size() * sizeof(BoxData), GL_DYNAMIC_DRAW);
    SSBO triangleSSBO(1, triangles.data(), triangles.size() * sizeof(TriangleData), GL_DYNAMIC_DRAW);
    SSBO circleSSBO(2, circles.data(), circles.size() * sizeof(CircleData), GL_DYNAMIC_DRAW);

    // Disable depth testing for 2D rendering
    glDisable(GL_DEPTH_TEST);

    // Initialize timing variables
    std::vector<float> frameRateHistory;
    float              rotationAngle = 0.0f;
    const float        rotationSpeed = 0.5f; // radians per second
    const float        orbitRadius   = 0.2f; // normalized screen units

    // Main render loop
    while (!glfwWindowShouldClose(window)) {
    	manager.UpdateSprite();
    	for (int count = 0; count < 7; ++count) {
    		circles[count].center = { (roundSpriteSet[count].boundingBox.left + roundSpriteSet[count].boundingBox.right) / 2.0 / 1440.f,
					(roundSpriteSet[count].boundingBox.top + roundSpriteSet[count].boundingBox.bottom) / 2.0  / 1440.f
				};
    	}

    	circleSSBO.UpdateData(circles.data(), circles.size() * sizeof(CircleData));

        // Create queries for timing measurement
        GLuint queryIDs[2];
        glGenQueries(2, queryIDs);

        // Start timing
        glQueryCounter(queryIDs[0], GL_TIMESTAMP);

        // Render scene
        sceneProgram.Use();
        sceneProgram.SetInt("boxCount", boxes.size());
        sceneProgram.SetInt("triangleCount", triangles.size());
        sceneProgram.SetInt("circleCount", circles.size());

        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        // End timing
        glQueryCounter(queryIDs[1], GL_TIMESTAMP);

        // Wait for query results
        GLint available = 0;
        while (!available) {
            glGetQueryObjectiv(queryIDs[1], GL_QUERY_RESULT_AVAILABLE, &available);
        }

        // Get timing results
        GLuint64 startTimeStamp, endTimeStamp;
        glGetQueryObjectui64v(queryIDs[0], GL_QUERY_RESULT, &startTimeStamp);
        glGetQueryObjectui64v(queryIDs[1], GL_QUERY_RESULT, &endTimeStamp);

        // Calculate frame time and FPS
        GLuint64 elapsedTime = endTimeStamp - startTimeStamp;
        double   elapsedMs   = static_cast<double>(elapsedTime) / 1000000.0;
        float    currentFPS  = 1000.0f / static_cast<float>(elapsedMs);

        // Update FPS history
        frameRateHistory.push_back(currentFPS);
        if (frameRateHistory.size() > 500) {
            frameRateHistory.erase(frameRateHistory.begin());
        }

        // Prepare ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Draw FPS chart
        DrawFPSChart(frameRateHistory);

        // Render ImGui
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // Swap buffers and poll events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    return 0;


	return 0;
}

/**
 * @brief Loads shader source code from a file
 * @param filename Path to the shader file
 * @return Shader source code as a string
 */
std::string LoadShaderSource(const char *filename) {
    // Open file in binary mode
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open shader file: " << filename << std::endl;
        return "";
    }

    // Get file size
    file.seekg(0, std::ios::end);
    std::streampos length = file.tellg();
    file.seekg(0, std::ios::beg);

    // Read file content
    std::string content(length, '\0');
    file.read(&content[0], length);

    // Check and skip UTF-8 BOM if present
    if (content.size() >= 3 &&
        static_cast<unsigned char>(content[0]) == 0xEF &&
        static_cast<unsigned char>(content[1]) == 0xBB &&
        static_cast<unsigned char>(content[2]) == 0xBF) {
        return content.substr(3);
    }

    return content;
}

/**
 * @brief Initializes GLFW window and OpenGL context
 * @return Pointer to the created window, or nullptr on failure
 */
GLFWwindow *InitializeWindow() {
    // Set OpenGL version and profile based on platform
#if defined(IMGUI_IMPL_OPENGL_ES2)
    // GL ES 2.0 + GLSL 100 (WebGL 1.0)
    const char *glsl_version = "#version 100";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
#elif defined(IMGUI_IMPL_OPENGL_ES3)
    // GL ES 3.0 + GLSL 300 es (WebGL 2.0)
    const char *glsl_version = "#version 300 es";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
#elif defined(__APPLE__)
    // GL 3.2 + GLSL 150
    const char *glsl_version = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#else
    // GL 3.0 + GLSL 130
    const char *glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
#endif

    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return nullptr;
    }

    // Create window
    GLFWwindow *window = glfwCreateWindow(1920, 1080, "PAR : Probabilistic Angular Radiance", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return nullptr;
    }

    glfwMakeContextCurrent(window);

    // Initialize GLAD
    if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        glfwTerminate();
        return nullptr;
    }

    // Initialize ImGui
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Set window resize callback
    glfwSetWindowSizeCallback(window, [](GLFWwindow *window, int width, int height) {
        glfwMakeContextCurrent(window);
        glViewport(0, 0, width, height);
    });

    return window;
}

/**
 * @brief Draws an FPS chart using ImGui
 * @param fpsHistory Vector containing FPS values history
 */
void DrawFPSChart(const std::vector<float> &fpsHistory) {
    ImGui::Begin("FPS Monitor");

    // Calculate average FPS
    float avgFPS = 0.0f;
    if (!fpsHistory.empty()) {
        for (float fps: fpsHistory) avgFPS += fps;
        avgFPS /= fpsHistory.size();
    }

    // Get current FPS
    float currentFPS = fpsHistory.empty() ? 0.0f : fpsHistory.back();

    // Display FPS information
    ImGui::Text("Current: %.1f FPS | Avg: %.1f FPS", currentFPS, avgFPS);

    // Calculate FPS range for chart
    float minFPS       = fpsHistory.empty() ? 0.0f : *std::min_element(fpsHistory.begin(), fpsHistory.end());
    float maxFPS       = fpsHistory.empty() ? 0.0f : *std::max_element(fpsHistory.begin(), fpsHistory.end());
    float rangePadding = std::max(10.0f, (maxFPS - minFPS) * 0.1f);

    // Create overlay text
    char overlay[32];
    snprintf(overlay, sizeof(overlay), "%.1f FPS", currentFPS);

    // Draw FPS chart
    ImGui::PlotLines(
        "##FPS_Chart",
        fpsHistory.data(),
        static_cast<int>(fpsHistory.size()),
        0,
        overlay,
        std::max(0.0f, minFPS - rangePadding),
        maxFPS + rangePadding,
        ImVec2(0, 150.0f)
    );

    // Display frame time
    float frameTime = 1000.0f / (currentFPS > 0 ? currentFPS : 1.0f);
    ImGui::Text("Frame time: %.2f ms", frameTime);

    ImGui::End();
}
