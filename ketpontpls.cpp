//=============================================================================================
// Szamitogepes grafika hazi feladat keret. Ervenyes 2017-tol.
// A //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// sorokon beluli reszben celszeru garazdalkodni, mert a tobbit ugyis toroljuk.
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kivéve
// - new operatort hivni a lefoglalt adat korrekt felszabaditasa nelkul
// - felesleges programsorokat a beadott programban hagyni
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL/GLUT fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Zsiga Tibor
// Neptun : L04P9O
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================

#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <vector>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>		// must be downloaded 
#include <GL/freeglut.h>	// must be downloaded unless you have an Apple
#endif


const unsigned int windowWidth = 600, windowHeight = 600;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// You are supposed to modify the code from here...

// OpenGL major and minor versions
int majorVersion = 3, minorVersion = 3;

void getErrorInfo(unsigned int handle) {
	int logLen;
	glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
	if (logLen > 0) {
		char * log = new char[logLen];
		int written;
		glGetShaderInfoLog(handle, logLen, &written, log);
		printf("Shader log:\n%s", log);
		delete log;
		}
	}

// check if shader could be compiled
void checkShader(unsigned int shader, char * message) {
	int OK;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
	if (!OK) {
		printf("%s!\n", message);
		getErrorInfo(shader);
		}
	}

// check if shader could be linked
void checkLinking(unsigned int program) {
	int OK;
	glGetProgramiv(program, GL_LINK_STATUS, &OK);
	if (!OK) {
		printf("Failed to link shader program!\n");
		getErrorInfo(program);
		}
	}

// vertex shader in GLSL
const char * vertexSource = R"(
	#version 330
    precision highp float;

	uniform mat4 MVP;			// Model-View-Projection matrix in row-major format

	layout(location = 0) in vec2 vertexPosition;	// Attrib Array 0
	layout(location = 1) in vec3 vertexColor;	    // Attrib Array 1
	out vec3 color;									// output attribute

	void main() {
		color = vertexColor;														// copy color from input to output
		gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1) * MVP; 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char * fragmentSource = R"(
	#version 330
    precision highp float;

	in vec3 color;				// variable input: interpolated color of vertex shader
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = vec4(color, 1); // extend RGB to RGBA
	}
)";

// row-major matrix 4x4
struct mat4 {
	float m[4][4];
public:
	mat4() {}
	mat4(float m00, float m01, float m02, float m03,
		 float m10, float m11, float m12, float m13,
		 float m20, float m21, float m22, float m23,
		 float m30, float m31, float m32, float m33) {
		m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
		m[1][0] = m10; m[1][1] = m11; m[1][2] = m12; m[1][3] = m13;
		m[2][0] = m20; m[2][1] = m21; m[2][2] = m22; m[2][3] = m23;
		m[3][0] = m30; m[3][1] = m31; m[3][2] = m32; m[3][3] = m33;
		}

	mat4 operator*(const mat4& right) {
		mat4 result;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				result.m[i][j] = 0;
				for (int k = 0; k < 4; k++) result.m[i][j] += m[i][k] * right.m[k][j];
				}
			}
		return result;
		}
	operator float*() { return &m[0][0]; }
	};

struct vec3 {
	float x, y, z;

	vec3(float x0 = 0, float y0 = 0, float z0 = 0) { x = x0; y = y0; z = z0; }

	vec3 operator*(float a) const { return vec3(x * a, y * a, z * a); }

	vec3 operator+(const vec3& v) const {
		return vec3(x + v.x, y + v.y, z + v.z);
		}
	vec3 operator-(const vec3& v) const {
		return vec3(x - v.x, y - v.y, z - v.z);
		}
	vec3 operator*(const vec3& v) const {
		return vec3(x * v.x, y * v.y, z * v.z);
		}	
	vec3 operator-() const {
		return vec3(-x, -y, -z);
		}
	vec3 normalize() const {
		return (*this) * (1 / (Length() + 0.000001));
		}
	float Length() const { return sqrtf(x * x + y * y + z * z); }

	operator float*() { return &x; }
	};

float dot(const vec3& v1, const vec3& v2) {
	return (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z);
	}

vec3 cross(const vec3& v1, const vec3& v2) {
	return vec3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
	}

// 3D point in homogeneous coordinates
struct vec4 {
	float v[4];

	vec4(vec3 vec) {
		vec4(vec.x, vec.y, vec.z, 1);
		}

	vec4(float x = 0, float y = 0, float z = 0, float w = 1) {
		v[0] = x; v[1] = y; v[2] = z; v[3] = w;
		}

	vec4 operator*(const mat4& mat) {
		vec4 result;
		for (int j = 0; j < 4; j++) {
			result.v[j] = 0;
			for (int i = 0; i < 4; i++) result.v[j] += v[i] * mat.m[i][j];
			}
		return result;
		}
	};

mat4 translateMtx(float x, float y, float z){
	return mat4(
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		x, y, z, 1
	);
}

class Camera{
public:
	vec3 wLookat = vec3(0.0f, 10.0f, 0.0f);
	vec3 wVup = vec3(0.0f, 1.0f, 0.0f);
	vec3 wEye = vec3(0.0f, 10.0f, 30.0f);

	float fov = M_PI / 4.0f; //45 fok
	float asp = windowWidth / windowHeight;
	float fp = 5.0f;
	float bp = 1500.0f;

	void zTranslate(float z){
		wLookat.z -= z;
		wEye.z -= z;
	}

	mat4 V(){
		vec3 w = (wEye - wLookat).normalize();
		vec3 u = cross(wVup, w).normalize();
		vec3 v = cross(w, u);

		vec3 tempVec = wEye * (-1.0f);

		return  translateMtx(tempVec.x, tempVec.y, tempVec.z) *
						mat4(u.x, v.x, w.x, 0.0f,
							 u.y, v.y, w.y, 0.0f,
							 u.z, v.z, w.z, 0.0f,
							 0.0f, 0.0f, 0.0f, 1.0f);
	}

	mat4 P(){
		float sy = 1.0f / tanf(fov/2.0f);
		return mat4(sy / asp, 0.0f, 0.0f, 0.0f,
					0.0f, sy, 0.0f, 0.0f,
					0.0f, 0.0f, -(fp + bp) / (bp - fp), -1.0f,
					0.0f, 0.0f, (-(2.0f)*fp*bp) / (bp - fp), 0.0f);
	}
};	

class Avatar{
public:
	Camera &cam;
	float sphereRadius = 50.0f;
	
	//lepeshez kellenek
	bool isAlive = true;	//ha mar meghalt, nem lepunk majd tobbet
	float stepIndex = 0;//ennyi egyseget kell lepni eppen
	float lastT = 0.0f;//utolso animate hivaskor ezt a parametert kaptuk


	Avatar(Camera &camera):cam(camera){}

	void move(){
		stepIndex += 20; //step 100 unit forward
	}

	void Animate(float t){
		if (lastT < 0.0001f)
			lastT = t;
		float timediff = t - lastT;

		lastT = t;
		if (stepIndex < 0.00001f) return;

		//nem biztos, hogy jo, ha a wEye-t toljuk el,
		//majd meglassuk (a vak is ezt mondta)
		cam.zTranslate(10.0f* timediff);


		stepIndex -= 10.0f*timediff;
	}
};
Camera camera;
Avatar avatar = Avatar(camera);
// handle of the shader program
unsigned int shaderProgram;

class CatmullRomSpline{
	std::vector<vec3> cps;//control points
	std::vector<float> ts;//param values
	vec3 Hermite(vec3 p0, vec3 v0, float t0,
				 vec3 p1, vec3 v1, float t1,
				 float t){
		//???
	}

public:
	void addControlPoint(vec3 cp, float t){
		cps.push_back(cp);
		ts.push_back(t);
	}

	vec3 r(float t){
		for(int i = 0; i < cps.size() - 1; i++){
			if (ts[i] <= t && t <= ts[i + 1])
				return Hermite(cps[i], vec3(i, i, 0), ts[i],
							   cps[i + 1], vec3(i + 1, 0, 0), ts[i + 1],
							   t);
		}
	}
	
};

class Snake{
	unsigned int vao;	// vertex array object id

						//pallo negy csucsanak koordinatai:
	vec3 balalso = vec3(-1.0f, -1.0f, 0);
	vec3 jobbalso = vec3(1.0f, -1.0f, 0);
	vec3 balfelso = vec3(-1.0f, 1.0f, 0);
	vec3 jobbfelso = vec3(1.0f, 1.0f, 0);

	float angle = -M_PI / 2.0f;
	float sx = 10.0f;
	float sy = 100.0f;
	float wTx = 0.0f;
	float wTy = 0.0f;
	float wTz = 0.0f;

public:
	void Create() {
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo[2];		// vertex buffer objects
		glGenBuffers(2, &vbo[0]);	// Generate 2 vertex buffer objects

									// vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]); // make it active, it is an array
		static float vertexCoords[] = { balalso.x, balalso.y, jobbalso.x, jobbalso.y, balfelso.x, balfelso.y,
			jobbalso.x, jobbalso.y, jobbfelso.x, jobbfelso.y, balfelso.x, balfelso.y };	// vertex data on the CPU
		glBufferData(GL_ARRAY_BUFFER,      // copy to the GPU
					 sizeof(vertexCoords), // number of the vbo in bytes
					 vertexCoords,		   // address of the data array on the CPU
					 GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
										   // Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
		glEnableVertexAttribArray(0);
		// Data organization of Attribute Array 0 
		glVertexAttribPointer(0,			// Attribute Array 0
							  2, GL_FLOAT,  // components/attribute, component type
							  GL_FALSE,		// not in fixed point format, do not normalized
							  0, NULL);     // stride and offset: it is tightly packed

											// vertex colors: vbo[1] -> Attrib Array 1 -> vertexColor of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]); // make it active, it is an array
		static float vertexColors[] = {
			0.64, 0.16, 0.16,
			0.64, 0.16, 0.16,
			0.64, 0.16, 0.16,
			0.64, 0.16, 0.16,
			0.64, 0.16, 0.16,
			0.64, 0.16, 0.16 };	// vertex data on the CPU
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexColors), vertexColors, GL_STATIC_DRAW);	// copy to the GPU

																							// Map Attribute Array 1 to the current bound vertex buffer (vbo[1])
		glEnableVertexAttribArray(1);  // Vertex position
									   // Data organization of Attribute Array 1
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL); // Attribute Array 1, components/attribute, component type, normalize?, tightly packed
		}
	void Draw() {
		mat4 Mscale(sx, 0, 0, 0,
					0, sy, 0, 0,
					0, 0, 1, 0,
					0, 0, 0, 1); // model matrix

		mat4 Mtranslate(1, 0, 0, 0,
						0, 1, 0, 0,
						0, 0, 1, 0,
						wTx, wTy, wTz, 1); // model matrix

		mat4 xRotate(1, 0, 0, 0,
					 0, cosf(angle), -sinf(angle), 0,
					 0, sinf(angle), cosf(angle), 0,
					 0, 0, 0, 1);

		mat4 MVPTransform = Mscale * xRotate * Mtranslate * camera.V() * camera.P();

		// set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
		int location = glGetUniformLocation(shaderProgram, "MVP");
		if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, MVPTransform); // set uniform variable MVP to the MVPTransform
		else printf("uniform MVP cannot be set\n");

		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		glDrawArrays(GL_TRIANGLES, 0, 6);	// draw a single triangle with vertices defined in vao
		}

};

class Pallo{
	unsigned int vao;	// vertex array object id

	//pallo negy csucsanak koordinatai:
	vec3 balalso  = vec3(-1.0f, -1.0f, 0);
	vec3 jobbalso = vec3( 1.0f, -1.0f, 0);
	vec3 balfelso = vec3(-1.0f,  1.0f, 0);
	vec3 jobbfelso= vec3( 1.0f,  1.0f, 0);

	float angle = -M_PI/2.0f;
	float sx = 10.0f;
	float sy = 100.0f;
	float wTx = 0.0f;
	float wTy = 0.0f;
	float wTz = 0.0f;

public:
	void Create() {
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo[2];		// vertex buffer objects
		glGenBuffers(2, &vbo[0]);	// Generate 2 vertex buffer objects

									// vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]); // make it active, it is an array
		static float vertexCoords[] = { balalso.x, balalso.y, jobbalso.x, jobbalso.y, balfelso.x, balfelso.y,
										jobbalso.x, jobbalso.y, jobbfelso.x, jobbfelso.y, balfelso.x, balfelso.y};	// vertex data on the CPU
		glBufferData(GL_ARRAY_BUFFER,      // copy to the GPU
					 sizeof(vertexCoords), // number of the vbo in bytes
					 vertexCoords,		   // address of the data array on the CPU
					 GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
										   // Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
		glEnableVertexAttribArray(0);
		// Data organization of Attribute Array 0 
		glVertexAttribPointer(0,			// Attribute Array 0
							  2, GL_FLOAT,  // components/attribute, component type
							  GL_FALSE,		// not in fixed point format, do not normalized
							  0, NULL);     // stride and offset: it is tightly packed

											// vertex colors: vbo[1] -> Attrib Array 1 -> vertexColor of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]); // make it active, it is an array
		static float vertexColors[] = { 
			0.64, 0.16, 0.16,
			0.64, 0.16, 0.16,
			0.64, 0.16, 0.16,
			0.64, 0.16, 0.16,
			0.64, 0.16, 0.16,
			0.64, 0.16, 0.16};	// vertex data on the CPU
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexColors), vertexColors, GL_STATIC_DRAW);	// copy to the GPU

																							// Map Attribute Array 1 to the current bound vertex buffer (vbo[1])
		glEnableVertexAttribArray(1);  // Vertex position
									   // Data organization of Attribute Array 1
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL); // Attribute Array 1, components/attribute, component type, normalize?, tightly packed
		}
	void Draw(){
		mat4 Mscale(sx, 0, 0, 0,
					0, sy, 0, 0,
					0, 0, 1, 0,
					0, 0, 0, 1); // model matrix

		mat4 Mtranslate(1, 0, 0, 0,
						0, 1, 0, 0,
						0, 0, 1, 0,
						wTx, wTy, wTz, 1); // model matrix

		mat4 xRotate(1, 0, 0, 0,
					 0, cosf(angle), -sinf(angle), 0,
					 0, sinf(angle), cosf(angle), 0,
					 0, 0, 0, 1);

		mat4 MVPTransform = Mscale * xRotate * Mtranslate * camera.V() * camera.P();

		// set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
		int location = glGetUniformLocation(shaderProgram, "MVP");
		if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, MVPTransform); // set uniform variable MVP to the MVPTransform
		else printf("uniform MVP cannot be set\n");

		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		glDrawArrays(GL_TRIANGLES, 0, 6);	// draw a single triangle with vertices defined in vao
	}
};

class Triangle {
public:
	unsigned int vao;	// vertex array object id
	float sx, sy;		// scaling
	float wTx, wTy, wTz;// translation

	Triangle() {
		Animate(0);
		}

	void Create() {
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo[2];		// vertex buffer objects
		glGenBuffers(2, &vbo[0]);	// Generate 2 vertex buffer objects

									// vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]); // make it active, it is an array
		static float vertexCoords[] = { -5, 0,  5, 0,  0, 8,66 };	// vertex data on the CPU
		glBufferData(GL_ARRAY_BUFFER,      // copy to the GPU
					 sizeof(vertexCoords), // number of the vbo in bytes
					 vertexCoords,		   // address of the data array on the CPU
					 GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
										   // Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
		glEnableVertexAttribArray(0);
		// Data organization of Attribute Array 0 
		glVertexAttribPointer(0,			// Attribute Array 0
							  2, GL_FLOAT,  // components/attribute, component type
							  GL_FALSE,		// not in fixed point format, do not normalized
							  0, NULL);     // stride and offset: it is tightly packed

											// vertex colors: vbo[1] -> Attrib Array 1 -> vertexColor of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]); // make it active, it is an array
		static float vertexColors[] = { 1, 0, 0,  0, 1, 0,  0, 0, 1 };	// vertex data on the CPU
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexColors), vertexColors, GL_STATIC_DRAW);	// copy to the GPU

																							// Map Attribute Array 1 to the current bound vertex buffer (vbo[1])
		glEnableVertexAttribArray(1);  // Vertex position
									   // Data organization of Attribute Array 1
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL); // Attribute Array 1, components/attribute, component type, normalize?, tightly packed
		}

	void Animate(float t) {
		sx = 1; // sinf(t);
		sy = 1; // cosf(t);
//		wTx = 4 * cosf(t / 2);
//		wTy = 4 * sinf(t / 2);
//		wTz = 10.0f;
		}

	void Draw() {
		mat4 Mscale(sx, 0, 0, 0,
					0, sy, 0, 0,
					0, 0, 1, 0,
					0, 0, 0, 1); // model matrix

		mat4 Mtranslate(1, 0, 0, 0,
						0, 1, 0, 0,
						0, 0, 1, 0,
						wTx, wTy, wTz, 1); // model matrix

		mat4 MVPTransform = Mscale * Mtranslate * camera.V() * camera.P();

		// set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
		int location = glGetUniformLocation(shaderProgram, "MVP");
		if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, MVPTransform); // set uniform variable MVP to the MVPTransform
		else printf("uniform MVP cannot be set\n");

		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		glDrawArrays(GL_TRIANGLES, 0, 3);	// draw a single triangle with vertices defined in vao
		}
	};

Triangle triangle1;
Triangle triangle2;
Pallo pallo;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	// Create objects by setting up their vertex data on the GPU
	triangle1.Create();

	triangle2.Create();


	pallo.Create();

	// Create vertex shader from string
	unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
	if (!vertexShader) {
		printf("Error in vertex shader creation\n");
		exit(1);
		}
	glShaderSource(vertexShader, 1, &vertexSource, NULL);
	glCompileShader(vertexShader);
	checkShader(vertexShader, "Vertex shader error");

	// Create fragment shader from string
	unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	if (!fragmentShader) {
		printf("Error in fragment shader creation\n");
		exit(1);
		}
	glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
	glCompileShader(fragmentShader);
	checkShader(fragmentShader, "Fragment shader error");

	// Attach shaders to a single program
	shaderProgram = glCreateProgram();
	if (!shaderProgram) {
		printf("Error in shader program creation\n");
		exit(1); 
		}
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);

	// Connect the fragmentColor to the frame buffer memory
	glBindFragDataLocation(shaderProgram, 0, "fragmentColor");	// fragmentColor goes to the frame buffer memory

																// program packaging
	glLinkProgram(shaderProgram);
	checkLinking(shaderProgram);
	// make this program run
	glUseProgram(shaderProgram);
	}

void onExit() {
	glDeleteProgram(shaderProgram);
	printf("exit");
	}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen

	triangle1.Draw();
	triangle2.Draw();
	pallo.Draw();
	
	
	glutSwapBuffers();									// exchange the two buffers
	}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
	if (key == ' ') {
		avatar.move();
		}
	}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
	

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	float sec = time / 1000.0f;				// convert msec to sec

	avatar.Animate(sec);
	triangle1.wTx = -10 - 4*cosf(2*sec);
	triangle1.wTy = 2 - 4*sinf(2*sec);
	triangle1.wTz = -15.0f;
	triangle1.Animate(sec);					// animate the triangle object
	triangle2.wTx = 10 + 4*cosf(2*sec);
	triangle2.wTy = 2 + 4*sinf(2*sec);
	triangle2.wTz = -50.0f;
	triangle2.Animate(sec);
	glutPostRedisplay();					// redraw the scene
	}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Do not touch the code below this line

int main(int argc, char * argv[]) {
	glutInit(&argc, argv);
#if !defined(__APPLE__)
	glutInitContextVersion(majorVersion, minorVersion);
#endif
	glutInitWindowSize(windowWidth, windowHeight);				// Application window is initially of resolution 600x600
	glutInitWindowPosition(100, 100);							// Relative location of the application window
#if defined(__APPLE__)
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_3_CORE_PROFILE);  // 8 bit R,G,B,A + double buffer + depth buffer
#else
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	glutCreateWindow(argv[0]);

#if !defined(__APPLE__)
	glewExperimental = true;	// magic
	glewInit();
#endif

	printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
	printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
	printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
	glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
	glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
	printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
	printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	onInitialization();

	glutDisplayFunc(onDisplay);                // Register event handlers
	glutMouseFunc(onMouse);
	glutIdleFunc(onIdle);
	glutKeyboardFunc(onKeyboard);
	glutKeyboardUpFunc(onKeyboardUp);
	glutMotionFunc(onMouseMotion);

	glutMainLoop();
	onExit();
	return 1;
	}
