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

//pontfenyforras
struct Light {
	vec3 position = vec3(30, 30, -70); //valami default ertek, talan jo is lesz
	float La = 300, Le = 1000;

	vec3 getLightDir(vec3 otherPos) { return otherPos - position; }
	vec3 getInRad(vec3 otherPos) {
		float dist = getDist(otherPos);
		dist *= dist;
		return Le - dist;
		}
	float getDist(vec3 otherPos) {
		return (otherPos - position).Length();
		}
	};

struct RenderState{
	mat4 M, Minv, V, P;
	vec3 wEye;
	Light light;

	vec3 kd, ks, ka;
	float shine;
};

class PerPixelShader{
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

	unsigned int shaderProg;

public:	

	PerPixelShader(){ Create(vertexSource, fragmentSource, "fragmentColor");	}

	void Create(const char *vsSrc, const char *fsSrc, const char *fsOutputName){
		unsigned int vs = glCreateShader(GL_VERTEX_SHADER);
		if (!vs) {
			printf("Error in vertex shader creation\n");
			exit(1);
			}

		glShaderSource(vs, 1, &vsSrc, NULL);
		glCompileShader(vs);
		checkShader(vs, "Vertex shader error");

		unsigned int fs = glCreateShader(GL_FRAGMENT_SHADER);
		if (!fs) {
			printf("Error in fragment shader creation\n");
			exit(1);
			}

		glShaderSource(fs, 1, &fsSrc, NULL);
		glCompileShader(fs);
		checkShader(fs, "Fragment shader error");

		shaderProg = glCreateProgram();
		if (!shaderProg) {
			printf("Error in shader program creation\n");
			exit(1);
			}

		glAttachShader(shaderProg, vs);
		glAttachShader(shaderProg, fs);

		glBindFragDataLocation(shaderProg, 0, fsOutputName);
		glLinkProgram(shaderProg);
	}
	void Bind(RenderState &state){
		glUseProgram(shaderProg);
		mat4 MVP = state.M * state.V * state.P;
		MVP.SetUniform(shaderProg, "MVP");
		state.M.SetUniform(shaderProg, "M");
		state.Minv.SetUniform(shaderProg, "Minv");
		
		int loc1 = glGetUniformLocation(shaderProg, "wLiPos");
		vec4 wLiPos = vec4(state.light.position);
		//nem lesz jó minden uniformmatrix4fv fuggvennyel sztem

		glUniformMatrix4fv(loc1, 1, GL_TRUE, &wLiPos.v[0]);

		int loc2 = glGetUniformLocation(shaderProg, "wEye");
		glUniformMatrix4fv(loc2, 1, GL_TRUE, state.wEye);

		int loc2 = glGetUniformLocation(shaderProg, "wEye");
		glUniformMatrix4fv(loc2, 1, GL_TRUE, state.wEye);


	}
};

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
	Camera *cam;
	float sphereRadius = 50.0f;
	
	//lepeshez kellenek
	bool isAlive = true;	//ha mar meghalt, nem lepunk majd tobbet
	float stepIndex = 0;//ennyi egyseget kell lepni eppen
	float lastT = 0.0f;//utolso animate hivaskor ezt a parametert kaptuk


	void move(){
		stepIndex += 20; //step 100 unit forward
	}

	void Animate(float t){
		if (lastT < 0.0001f)
			lastT = t;
		float timediff = t - lastT;

		lastT = t;
		if (stepIndex < 0.00001f) return;

		cam.zTranslate(10.0f* timediff);

		stepIndex -= 10.0f*timediff;
	}
};
Camera camera;

// handle of the shader program
unsigned int shaderProgram;

struct Material{
	vec3 kd = 5, ks = 5, ka = 5;
	float shininess = 5;
};

struct Geometry{
	unsigned int vao, nVtx = 0;

	float sX = 1, sY = 1, sZ = 1;	//skalazas
	float tX = 0, tY = 0, tZ = 0;	//eltolas
	float xAngle = 0;				//forgatas x korul
	float zAngle = 0;				//forgatas z korul

	Geometry(){
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
	}

	void Draw(){
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		glDrawArrays(GL_TRIANGLES, 0, nVtx);	// draw a single triangle with vertices defined in vao
	}

	void Animate(float t){};
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

class Sphere: public ParamSurface{
	vec3 center;
	float radius;

	VertexData genVertexData(float u, float v)override{
		VertexData data;

		data.normal = vec3(cos(u * 2 * M_PI)*sin(v * M_PI),
						   sin(u * 2 * M_PI)*sin(v * M_PI),
						   cos(v*M_PI));
		data.position = data.normal * radius + center;
		data.u = u;
		data.v = v;

		return data;
	}
public:
	Sphere(vec3 c, float r):ParamSurface(), center(c), radius(r){
		Create(16, 8);
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

class Snake: public ParamSurface{

	CatmullRomSpline gerincgorbe;	//kigyo gerincgorbeje

	const float maxRadius = 2.0f;
	const float neckRadius = 1.0f;

	float animationOffset = 0;

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

	vec3 getSurfacePoint(vec3 center, float angleInDegrees, float radius) {
		float angleInRadian = angleInDegrees * (M_PI / 180);
		float x = center.x + radius * cosf(angleInRadian);
		float y = center.y;
		float z = center.z + radius * sinf(angleInRadian);

		return vec3(x, y, z);
		}

	VertexData genVertexData(float t, float angle)override{//itt a szog 360-nal osztva van a create fv miatt, majd vissza kell szorozni
		vec3 center = gerincgorbe.r(t);
		float sugar = getRadius(t);

		float angle360 = angle * 360.0f;

		VertexData data;
		data.position = getSurfacePoint(center, angle360, sugar);
		data.normal = (data.position - center).normalize();
		data.u = t;
		data.v = angle;

		return data;
	}

public:
	Snake(float animoffset = 0):ParamSurface(),animationOffset(animoffset){
		gerincgorbe = CatmullRomSpline();
		tX = 5; tY = 10; tZ = -30;
		zAngle = M_PI_4;

		Create(50, 360);
	}

	void Animate(float t){
		gerincgorbe.Animate(t);	
		
		Create(50, 360);
	}
};

class Homok: public ParamSurface{
	float getHeight(const float x,const float y){
		return 5.0f* cosf(x) + 2.0f*sinf(y);
	}

	//kiszamolja a normalvektort
	vec3 getNormal(float x, float y){
		vec3 dx = -5.0f * sinf(x);
		vec3 dy =  2.0f * cosf(y);

		return dot(dx, dy);
	}

	VertexData genVertexData(float u, float v)override{
		vec3 pos = vec3(u,v, getHeight(u,v));
		vec3 norm = getNormal(u, v);

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

		Create(50, 50);
	}
};

class Pallo : public ParamSurface{
	
	VertexData genVertexData(float u, float v)override{
		VertexData data;

		data.position = vec3(u, v, 0);
		data.normal = vec3(0, 0, 1);
		data.u = u;
		data.v = v;

		return data;
	}

public:
	Pallo():ParamSurface(){
		sX = 10.0f; sY = 100.0f; sZ = 1;
		float tX = -50; tY = 0; tZ = 0;
		xAngle = -M_PI_2;

		Create(100, 100);
	}
};

class Object{
public:
	Material *material;
	Geometry *geometry;
	PerPixelShader *shader;


	void Animate(float t){
		geometry->Animate(t);
	}
	
	void Draw(RenderState state){
		state.M = scale(geometry->sX, geometry->sY, geometry->sZ)* xRotate(geometry->xAngle)*
			zRotate(geometry->zAngle)* translate(geometry->tX, geometry->tY, geometry->tZ);
		
		state.Minv = translate(-geometry->tX, -geometry->tY, -geometry->tZ) *
			zRotate(-geometry->zAngle) * xRotate(-geometry->xAngle) *
			scale(1.0f / geometry->sX, 1.0f / geometry->sY, 1.0f / geometry->sZ);

		shader->Bind(state);
		geometry->Draw();
	}
};



class Scene{
	Light light;
	
public:
	Avatar avatar;
	std::vector<Object*> objects;
	
	void Render(){
		RenderState state;
		state.wEye = avatar.cam->wEye;
		state.V = avatar.cam->V();
		state.P = avatar.cam->P();
		state.light = light;

		for (Object* obj : objects)
			obj->Draw(state);
	}

	void Animate(float t){
		avatar.Animate(t);

		for (Object* obj : objects)
			obj->Animate(t);
	}

	~Scene(){
		for (Object* obj : objects)
			delete obj;
	}
};

Scene scene;




// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	// Create objects by setting up their vertex data on the GPU

	Object *homok = new Object();
	homok->geometry = new Homok();
	homok->material = new Material();
	homok->shader = new PerPixelShader();
	scene.objects.push_back(homok);

	Object *pallo = new Object();
	pallo->geometry = new Pallo();
	pallo->material = new Material();
	pallo->shader = new PerPixelShader();
	scene.objects.push_back(pallo);

	Object *snek1 = new Object();
	snek1->geometry = new Snake();
	snek1->material = new Material();
	snek1->shader = new PerPixelShader();
	scene.objects.push_back(snek1);

	Object *snek2 = new Object();
	snek2->geometry = new Snake(1.0f);
	snek2->material = new Material();
	snek2->shader = new PerPixelShader();
	snek2->geometry->zAngle *= -1.0f;
	snek2->geometry->tX *= -1.0f;
	snek2->geometry->tZ = -60;
	scene.objects.push_back(snek2);

	Object *snek3 = new Object();
	snek3->geometry = new Snake(2.0f);
	snek3->material = new Material();
	snek3->shader = new PerPixelShader();
	snek3->geometry->tZ = -100;
	scene.objects.push_back(snek3);

	}

void onExit() {
	glDeleteProgram(shaderProgram);
	printf("exit");
	}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0.9, 0.9, 0);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
	
	scene.Render();

	glutSwapBuffers();									// exchange the two buffers
	}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
	
	if (key == ' ') {
		scene.avatar.move();
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

	scene.Animate(sec);

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

	glDisable(GL_CULL_FACE);				   //backface culling is off

	glutMainLoop();
	onExit();
	return 1;
	}
