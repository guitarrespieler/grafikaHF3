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

/*
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
)";*/

const char *vertexSource = R"(
uniform mat4 M, Minv, MVP;
uniform vec4 wLiPos;
uniform vec3 wEye;

layout(location = 0) in vec3 vtxPos;	//pos in model sp
layout(location = 1) in vec3 vtxNorm;	//normal in mod sp

out vec3 wNormal;	//normal in world space
out vec3 wView;		//view in world space
out vec3 wLight;	//light dir in world space

void main() {
	gl_Position = vec4(vtxPos, 1) * MVP;
	
	wLight  = wLiPos.xyz * wPos.w - wPos.xyz * wLiPos.w;
	wView   = wEye * wPos.w - wPos.xyz;
	wNormal = (Minv - vec4(vtxNorm, 0)).xyz;
})";

const char *fragmentSource = R"(
uniform vec3 kd, ks, ka;	//diffuse, specular, ambient ref
uniform vec3 La, Le;		//ambient and point source rad
uniform float shine;		//shinines for specular ref

in vec3 wNormal;			//interpolated world sp normal
in vec3 wView;				//interpolated world sp view
in vec3 wLight;				//interpolated world sp illum ref
out vec4 fragmentColor;		//output goes to frame buffer

void main() {
	vec3 N = normalize(wNormal);
	vec3 V = normalize(wView);
	vec3 L = normalize(wLight);
	vec3 H = normalize(L + V);

	float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
	vec3 color = ka * La +
				(kd * cost + ks * pow(cosd, shine)) * Le;
	fragmentColor = vec4(color, 1);
})";

/*
// fragment shader in GLSL
const char * fragmentSource = R"(
	#version 330
    precision highp float;

	in vec3 color;				// variable input: interpolated color of vertex shader
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = vec4(color, 1); // extend RGB to RGBA
	}
)";*/

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

	void SetUniform(unsigned shaderProg, char *name){
		int loc = glGetUniformLocation(shaderProg, name);
		glUniformMatrix4fv(loc, 1, GL_TRUE, &m[0][0]);
	}
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

mat4 translate(float x, float y, float z){
	return mat4(1, 0, 0, 0,
				0, 1, 0, 0,
				0, 0, 1, 0,
				x, y, z, 1);
}
mat4 scale(float x, float y, float z){
	return mat4(x, 0, 0, 0,
				0, y, 0, 0,
				0, 0, z, 0,
				0, 0, 0, 1);
}
mat4 xRotate(float xAngle) {
	return mat4(1, 0, 0, 0,
				0, cosf(xAngle), -sinf(xAngle), 0,
				0, sinf(xAngle), cosf(xAngle), 0,
				0, 0, 0, 1);
}
mat4 zRotate(float zAngle) {
	return mat4(cosf(zAngle), -sinf(zAngle), 0, 0,
				sinf(zAngle), cosf(zAngle), 0, 0,
				0, 0, 1, 0,
				0, 0, 0, 1);
}

class Camera{
public:
	vec3 wLookat = vec3(0.0f, 10.0f, 0.0f);
	vec3 wVup = vec3(0.0f, 1.0f, 0.0f);
	vec3 wEye = vec3(0.0f, 15.0f, 55.0f);

	float fov = M_PI_4; //45 fok
	float asp = windowWidth / windowHeight;
	float fp = 5.0f;
	float bp = 1500.0f;

	void zTranslate(float z){
		wLookat.z -= z; //kivonjuk, mert -z fele nez a kamera.
		wEye.z -= z;    //szinten.
	}

	mat4 V(){
		vec3 w = (wEye - wLookat).normalize();
		vec3 u = cross(wVup, w).normalize();
		vec3 v = cross(w, u);

		vec3 tempVec = wEye * (-1.0f);

		return  translate(tempVec.x, tempVec.y, tempVec.z) *
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

struct Geometry{
	unsigned int vao, nVtx;

	float sX = 1, sY = 1, sZ = 1;	//skalazas
	float tX = 0, tY = 0, tZ = 0;	//eltolas
	float xAngle = 0;				//forgatas x korul
	float zAngle = 0;				//forgatas z korul

	void Create(){
		glGenVertexArrays(1,&vao); 
		glBindVertexArray(vao);
	}

	void Draw(){
		mat4 M = scale(sX, sY, sZ) *xRotate(xAngle) * zRotate(zAngle) * translate(tX, tY, tZ);

		mat4 Minv = translate(-tX, -tY, -tZ) * zRotate(-zAngle) * xRotate(-xAngle) * scale(1.0f / sX, 1.0f / sY, 1.0f / sZ);

		mat4 MVP = M * camera.V() * camera.P();

		M.SetUniform(shaderProgram, "M");
		Minv.SetUniform(shaderProgram, "Minv");
		MVP.SetUniform(shaderProgram, "MVP");

		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		glDrawArrays(GL_TRIANGLES, 0, nVtx);	// draw a single triangle with vertices defined in vao
	}
};

struct VertexData{
	vec3 position, normal;
	float u, v;
};

struct ParamSurface : Geometry{
	ParamSurface():Geometry(){}
	virtual VertexData genVertexData(float u, float v) = 0;
	
	void Create(int N, int M){
		nVtx = N * M * 6;		
		unsigned int vbo;

		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);

		VertexData *vtxData = new VertexData[nVtx], *pVtx = vtxData;

		for (int i = 0; i < N; i++) {
			for(int j = 0; j < M; j++){
				//egyik haromszog
				*pVtx++ = genVertexData((float)i / N, (float)j / M);
				*pVtx++ = genVertexData((float)(i+1) / N, (float)j / M);
				*pVtx++ = genVertexData((float)i / N, (float)(j+1) / M);

				//masik haromszog
				*pVtx++ = genVertexData((float)(i+1) / N, (float)j / M);
				*pVtx++ = genVertexData((float)i / N, (float)(j+1) / M);
				*pVtx++ = genVertexData((float)(i+1) / N, (float)(j+1) / M);
			}
		}
		glBufferData(GL_ARRAY_BUFFER, nVtx * sizeof(VertexData), vtxData, GL_STATIC_DRAW);
		
		glEnableVertexAttribArray(0);		//attribArray 0 = POSITION
		glEnableVertexAttribArray(1);		//attribArray 1 = NORMALVEC
		glEnableVertexAttribArray(2);		//attribArray 2 = UV

		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, u));

		//felszabaditjuk, mar nem kell
		delete[] vtxData;
	}
};

class CatmullRomSpline{
	vec3 startVelocity = vec3(-1, 0.5, 0); //kezdo sebessegvektor
	vec3 endVelocity = vec3(0, -1, 0); //utolso sebessegvektor

	vec3 Hermite(vec3 p0, vec3 v0, float t0,
				 vec3 p1, vec3 v1, float t1,
				 float t){
		
		vec3 a0 = p0;
		vec3 a1 = v0;
		vec3 a2 = ((p1 - p0)* 3.0f * (1.0f / ((t1 - t0)*(t1 - t0)))) -
			((v1 + v0*2.0f) * (1.0f/(t1 - t0)));
		vec3 a3 = (((p0 - p1) * 2.0f) * (1.0f / ((t1 - t0)*(t1 - t0)*(t1 - t0)))) +
			((v1 + v0)* (1.0f/((t1 - t0)*(t1 - t0))));

		vec3 r = a3*(t - t0)*(t - t0)*(t - t0) +
			a2*(t - t0)*(t - t0) + a1*(t - t0) + a0;

		return r;
	}

	//megadja az i-edik sebessegvektort
	vec3 getVi(int i){
		if (i == 0)
			return startVelocity;
		if (i == (cps.size() - 1))
			return endVelocity;

		vec3 szamlalo1 = cps[i + 1] - cps[i];//r(i+1) - r(i)
		float nevezo1 = ts[i + 1] - ts[i];   //t(i+1) - t(i)
		vec3 hanyados1 = szamlalo1 * (1.0f / nevezo1);

		vec3 szamlalo2 = cps[i] - cps[i - 1];//r(i) - r(i-1)
		float nevezo2 = ts[i] - ts[i - 1];	 // t(i)- t(i-1)
		vec3 hanyados2 = szamlalo2 * (1.0f / nevezo2);

		vec3 vi = (hanyados1 + hanyados2) * 0.5f;

		return vi;
	}

	void addControlPoint(vec3 cp, float t){
		cps.push_back(cp);
		ts.push_back(t);		
	}

public:
	std::vector<vec3> cps;//control points
	std::vector<float> ts;//param values

	CatmullRomSpline(){
		vec3 p1 = vec3(0, 40, 0);
		vec3 p2 = vec3(-8, 30, 0);
		vec3 p3 = vec3(4, 20, 0);
		vec3 p4 = vec3(-8, 10, 0);
		vec3 p5 = vec3(0, 0, 0);

		addControlPoint(p1, 0.0f);
		addControlPoint(p2, 1.0f);
		addControlPoint(p3, 2.0f);
		addControlPoint(p4, 3.0f);
		addControlPoint(p5, 4.0f);
	}

	vec3 r(float t){
		for(int i = 0; i < cps.size() - 1; i++){
			if (ts[i] <= t && t <= ts[i + 1]){
				vec3 vi = getVi(i);
				vec3 viPlusOne = getVi(i + 1);
				return Hermite(cps[i], vi, ts[i],
							   cps[i + 1], viPlusOne, ts[i + 1],
							   t);
			}		
		}
	}

	void Animate(float t){
		if (cps.empty())
			return;

		for(int i = 1; i < cps.size(); i++){
			vec3 vec = cps[i];
			vec.x = 4.0f * cosf(2.5f * t);
			if (i % 2 == 0)
				vec.x *= -1.0f;
			cps[i] = vec;
		}		
	}
};

class Triangle {
public:
	unsigned int vao;	// vertex array object id
	float sX = 1, sY = 1, sZ = 1;		// scaling
	float tX = 0, tY = 0, tZ = 0;// translation
	float zAngle = 0.0f;					//z rotate

	vec3 A = vec3(-5, 0, 0);
	vec3 B = vec3(5, 0, 0);
	vec3 C = vec3(0, 8.66, 0);

	vec3 colorA = vec3(1, 0, 0);
	vec3 colorB = vec3(0, 1, 0);
	vec3 colorC = vec3(0, 0, 1);

	void Create() {
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo[2];		// vertex buffer objects
		glGenBuffers(2, &vbo[0]);	// Generate 2 vertex buffer objects

									// vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]); // make it active, it is an array
		static float vertexCoords[] = { A.x, A.y, A.z, B.x, B.y, B.z, C.x, C.y, C.z};	// vertex data on the CPU
		glBufferData(GL_ARRAY_BUFFER,      // copy to the GPU
					 sizeof(vertexCoords), // number of the vbo in bytes
					 vertexCoords,		   // address of the data array on the CPU
					 GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
										   // Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
		glEnableVertexAttribArray(0);
		// Data organization of Attribute Array 0 
		glVertexAttribPointer(0,			// Attribute Array 0
							  3, GL_FLOAT,  // components/attribute, component type
							  GL_FALSE,		// not in fixed point format, do not normalized
							  0, NULL);     // stride and offset: it is tightly packed

											// vertex colors: vbo[1] -> Attrib Array 1 -> vertexColor of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]); // make it active, it is an array
		static float vertexColors[] = {
			colorA.x, colorA.y, colorA.z,
			colorB.x, colorB.y, colorB.z,
			colorC.x, colorC.y, colorC.z };	// vertex data on the CPU
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexColors), vertexColors, GL_STATIC_DRAW);	// copy to the GPU

																							// Map Attribute Array 1 to the current bound vertex buffer (vbo[1])
		glEnableVertexAttribArray(1);  // Vertex position
									   // Data organization of Attribute Array 1
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL); // Attribute Array 1, components/attribute, component type, normalize?, tightly packed
		}


	void Draw() {
		mat4 MVPTransform = scale(sX,sY,sZ) * zRotate(zAngle) * translate(tX, tY, tZ) * camera.V() * camera.P();

		// set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
		int location = glGetUniformLocation(shaderProgram, "MVP");
		if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, MVPTransform); // set uniform variable MVP to the MVPTransform
		else printf("uniform MVP cannot be set\n");

		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		glDrawArrays(GL_TRIANGLES, 0, 3);	// draw a single triangle with vertices defined in vao
		}
	};

class Snake{
	GLuint vao, vbo;        // vertex array object, vertex buffer object	
	std::vector<float> trianglesData;//itt tartom a kigyot alkoto haromszogek vertexeit es fragmentjeit
	
	float maxRadius = 2.0f;
	float neckRadius = 1.0f;

	//megadja a profilgorbe sugarat
	//parameter a gorbe ts parameterei menten fog menni
	float getRadius(float t){
		if (fabs(t - gerincgorbe.ts[0]) < 0.000001)
			return gerincgorbe.ts[0];

		if(gerincgorbe.ts[0] < t && gerincgorbe.ts[2] > t){
			return t + 0.001f;
		}

		int size = gerincgorbe.ts.size();
		float utolso = gerincgorbe.ts[size - 1];
		float utolsoelotti = gerincgorbe.ts[size - 2];

		if(utolso > t && utolsoelotti < t){
			float kulonbseg = fabs(utolso - t);

			float rad = neckRadius + kulonbseg;

			if (rad < maxRadius)
				return rad;
		}
		return maxRadius;
	}


	//letrehozza a kigyo testet alkoto haromszogeket
	void generateTriangles(){
		trianglesData.clear();
		int tsSize = gerincgorbe.ts.size();

		float tNovekmeny = 0.1f;
		float szogNovekmeny = 30.0f;

		for(float t = 0.0f; t < gerincgorbe.ts[tsSize - 1] - tNovekmeny; t += tNovekmeny){
			vec3 center1 = gerincgorbe.r(t);							//ez lesz a kor kozepe
			vec3 center2 = gerincgorbe.r(t + tNovekmeny);				//ez a kovetkezonek a kozepe
			float sugar1 = getRadius(t);								//ez lesz a kor sugara
			float sugar2 = getRadius(t + tNovekmeny);					//ez a kovetkezonek a sugara

			for(float szog = 0.0f; szog <= 360; szog+= szogNovekmeny){	
				vec3 A = getSurfacePoint(center1, szog, sugar1);
				vec3 B = getSurfacePoint(center1, szog + szogNovekmeny, sugar1);
				vec3 C = getSurfacePoint(center2, szog, sugar2);
				vec3 D = getSurfacePoint(center2, szog + szogNovekmeny, sugar2);

				//vec3 colorA = getSurfacePointColor(center1, szog, sugar1);
				//vec3 colorB = getSurfacePointColor(center1, szog + szogNovekmeny, sugar1);
				//vec3 colorC = getSurfacePointColor(center2, szog, sugar2);
				//vec3 colorD = getSurfacePointColor(center2, szog + szogNovekmeny, sugar2);
				
				vec3 colorA = vec3(0, 0, 0);
				vec3 colorB = vec3(0, 0.8, 0.1);
				vec3 colorC = vec3(0, 0.8, 0.1);
				vec3 colorD = vec3(0,0,0);

				writeVertexDataToVector(A, B, C,colorA,colorB,colorC);
				writeVertexDataToVector(B, C, D,colorB,colorC,colorD);
			}
		}
	}

	vec3 getSurfacePoint(vec3 center, float angleInDegrees, float radius){
		float angleInRadian = angleInDegrees * (M_PI / 180);
		float x = center.x + radius * cosf(angleInRadian);
		float y = center.y;
		float z = center.z + radius * sinf(angleInRadian);

		return vec3(x,y,z);
	}

	void writeVertexDataToVector(vec3 A, vec3 B, vec3 C,vec3 colorA, vec3 colorB, vec3 colorC) {
		trianglesData.push_back(A.x);
		trianglesData.push_back(A.y);
		trianglesData.push_back(A.z);
		trianglesData.push_back(B.x);
		trianglesData.push_back(B.y);
		trianglesData.push_back(B.z);
		trianglesData.push_back(C.x);
		trianglesData.push_back(C.y);
		trianglesData.push_back(C.z);
		trianglesData.push_back(colorA.x);
		trianglesData.push_back(colorA.y);
		trianglesData.push_back(colorA.z);
		trianglesData.push_back(colorB.x);
		trianglesData.push_back(colorB.y);
		trianglesData.push_back(colorB.z);
		trianglesData.push_back(colorC.x);
		trianglesData.push_back(colorC.y);
		trianglesData.push_back(colorC.z);
	}
public:
	CatmullRomSpline gerincgorbe = CatmullRomSpline();			//kigyo gerincgorbeje
	int tX = 5; int tY = 10; int tZ = -30;						//eltolas
	float zAngle = M_PI_4;										//z koruli elforgatas szoge

	void Create(){
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0
		glEnableVertexAttribArray(1);  // attribute array 1

									   // Map attribute array 0 to the vertex data of the interleaved vbo
		glVertexAttribPointer(0,							//attribute array
							  3,							//components/attribute
							  GL_FLOAT,					//component type
							  GL_FALSE,					//normalize?
							  3 * sizeof(float),			//stride
							  reinterpret_cast<void*>(0));//offset

														  // Map attribute array 1 to the color data of the interleaved vbo
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), reinterpret_cast<void*>(9 * sizeof(float)));
	}

	void Draw(){	
		if (trianglesData.empty())
			return;

		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		// copy data to the GPU
		glBufferData(GL_ARRAY_BUFFER, trianglesData.size() * sizeof(float), &trianglesData[0], GL_DYNAMIC_DRAW);


		mat4 MVPTransform = zRotate(zAngle) * translate(tX,tY,tZ) * camera.V() * camera.P();

		// set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
		int location = glGetUniformLocation(shaderProgram, "MVP");
		if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, MVPTransform); // set uniform variable MVP to the MVPTransform
		else printf("uniform MVP cannot be set\n");

		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		glDrawArrays(GL_TRIANGLES, 0, trianglesData.size());	// draw a single triangle with vertices defined in vao
	}

	void Animate(float t){
		gerincgorbe.Animate(t);	
		generateTriangles();
	}
};

class Homok: public ParamSurface{
	float getHeight(const float x,const float y){
		return 5.0f* cosf(x) + 2.0f*sinf(y);
	}

	VertexData genVertexData(float u, float v){
		vec3 pos = vec3(u,v, getHeight(u,v));
		vec3 norm = vec3(0,1,0);
		VertexData data = VertexData();
		data.position = pos;
		data.normal = norm;
		data.u = u;
		data.v = v;

		return data;		
	}

public:
	Homok():ParamSurface(){
		sX = 50; sY = 10; sZ = 1;
		tX = -300; tY = -30; tZ = 50;
		float xAgle = M_PI_2;
	}

	void Create(){
		Geometry::Create();

		ParamSurface::Create(50, 50);
	}
};

class Pallo{
	unsigned int vao;	// vertex array object id

	//pallo negy csucsanak koordinatai:
	vec3 balalso  = vec3(-1.0f, -1.0f, 0);
	vec3 jobbalso = vec3( 1.0f, -1.0f, 0);
	vec3 balfelso = vec3(-1.0f,  1.0f, 0);
	vec3 jobbfelso= vec3( 1.0f,  1.0f, 0);

	float angle = -M_PI_2;
	float sX = 10.0f, sY = 100.0f, sZ = 1;
	float tX = 0, tY = 0, tZ = 0;

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
		mat4 MVPTransform = scale(sX,sY,sZ) * xRotate(angle) * translate(tX, tY, tZ) * camera.V() * camera.P();

		// set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
		int location = glGetUniformLocation(shaderProgram, "MVP");
		if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, MVPTransform); // set uniform variable MVP to the MVPTransform
		else printf("uniform MVP cannot be set\n");

		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		glDrawArrays(GL_TRIANGLES, 0, 6);	// draw two triangles with vertices defined in vao
	}
};



Triangle triangle1;
Triangle triangle2;
Pallo pallo;
Homok homok;

Snake snek1;
Snake snek2;
Snake snek3;


// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	// Create objects by setting up their vertex data on the GPU
	homok.Create();

	triangle1.Create();
	triangle2.Create();

	pallo.Create();

	snek2.zAngle *= -1.0f;
	snek2.tX *= -1.0f;
	snek2.tZ = -60;

	snek3.tZ = -100;

	snek1.Create();
	snek2.Create();
	snek3.Create();

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
	glClearColor(0, 0.9, 0.9, 0);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
	homok.Draw();

	triangle1.Draw();
	triangle2.Draw();
	pallo.Draw();

	snek3.Draw();
	snek2.Draw();
	snek1.Draw();
	
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
	triangle1.tX = -10 - 4*cosf(2*sec);
	triangle1.tY = 2 - 4*sinf(2*sec);
	triangle1.tZ = -15.0f;
	triangle2.tX = 10 + 4*cosf(2*sec);
	triangle2.tY = 2 + 4*sinf(2*sec);
	triangle2.tZ = -50.0f;

	snek1.Animate(sec);
	snek2.Animate(sec + 1);				//eltolom kicsit a frekvenciat
	snek3.Animate(sec + 2);

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
