#include "Main.h"
#include "util.h"
#include <math.h>
#include <Windows.h>
#include <math.h>

// OpenCV includes
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <iostream>
#include <ctype.h>

using namespace cv;

float gridL = 6;

//inicializadores
GLfloat mouse_x, mouse_y;

//constantes
const double translateCameraConst = .1;
const double rotateCameraConst = 0.03;
const double rotateCameraMouseConst = 0.003;

bool buffer[250];

double frustrumTop, frustrumBottom, frustrumLeft, frustrumRight;
double g_Width, g_Height;
double zfar = 100.0f;
double znear = 0.1f;
GLdouble fovy = 45;

double focalDistance = 1;
double mousex, mousey;
double deltax, deltay;
dMatrix KMdMatrix(4, dVector(4));
dMatrix projM = dMatrix(4, dVector(4));
dMatrix extM = dMatrix(4, dVector(4));
dMatrix intrM = dMatrix(4, dVector(4));

GLfloat  K[9], projMatrix[16], extrinsic[16], intrisic[16];

dMatrix Ext = dMatrix(4, dVector(4));

void FimDoPrograma()
{
	exit(1);
}

cv::VideoCapture cap;
Mat frame;

void initCV()
{
	cap.open("Resources\\InputData\\video2.mp4");

	if (!cap.isOpened())
	{
		cout << "error, could not open the capture" << endl;
		system("pause");
		exit(1);
	}

	namedWindow("video", WINDOW_AUTOSIZE);
}

void initialize()
{
	initCV();

	glClearColor(0.0f, 0.0f, 0.0f, 0.5f);				// Black Background
	glClearDepth(1.0f);									// Depth Buffer Setup
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);							// Enables Depth Testing
	glShadeModel(GL_SMOOTH);							// Enable Smooth Shading

	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);	// Really Nice Perspective Calculations

	//inicializa a matriz extrinsica;
	Ext[0][0] = 1; Ext[1][1] = 1; Ext[2][2] = -1; Ext[3][3] = 1;
}

void cameraTranslate(double ctx, double cty, double ctz)
{
	Ext[0][3] += ctx;
	Ext[1][3] += cty;
	Ext[2][3] += ctz;
}

/*
angle em rad
*/
void cameraRotateY(double angle)
{
	dMatrix roty(4, dVector(4));
	dMatrix iNeg(4, dVector(4));

	roty[0][0] = cos(angle);
	roty[0][2] = sin(angle);
	roty[1][1] = 1;
	roty[2][0] = -sin(angle);
	roty[2][2] = cos(angle);
	roty[3][3] = 1;

	iNeg[0][0] = -1;
	iNeg[0][1] = 0;
	iNeg[0][2] = 0;
	iNeg[0][3] = 0;

	iNeg[1][0] = 0;
	iNeg[1][1] = -1;
	iNeg[1][2] = 0;
	iNeg[1][3] = 0;
	iNeg[2][0] = 0;

	iNeg[2][0] = 0;
	iNeg[2][1] = 0;
	iNeg[2][2] = -1;
	iNeg[2][3] = 0;

	iNeg[3][0] = 0;
	iNeg[3][1] = 0;
	iNeg[3][2] = 0;
	iNeg[3][3] = -1;

	dVector T(4);
	T[0] = Ext[0][3];
	T[1] = Ext[1][3];
	T[2] = Ext[2][3];
	T[3] = 1;

	dMatrix R = getRotationNN(Ext);
	dMatrix Rt = transpose(R);
	dVector C = multiplicacaoN1(Rt, T);
	C = multiplicacaoN1(iNeg, C);

	R = multiplicacaoNN(R, roty);
	T = multiplicacaoN1(R, C);
	T = multiplicacaoN1(iNeg, T);
	Ext = R;
	Ext[0][3] = T[0];
	Ext[1][3] = T[1];
	Ext[2][3] = T[2];
}



void normalizeCamera()
{
	double tx = Ext[0][3], ty = Ext[1][3], tz = Ext[2][3];

	Ext[0][3] = 0; Ext[1][3] = 0; Ext[2][3] = 0;

	for (int i = 0; i < 4; i++)
	{
		Ext[i] = normalize(Ext[i]);
	}

	Ext[0][3] = tx; Ext[1][3] = ty; Ext[2][3] = tz;
}

void myreshape(GLsizei w, GLsizei h)
{
	g_Width = w;
	g_Height = h;

	glViewport(0, 0, g_Width, g_Height);

	frustrumTop = tan(fovy * 3.14159 / 360) * 0.1;
	frustrumBottom = -frustrumTop;
	frustrumLeft = g_Width / g_Width * frustrumBottom;
	frustrumRight = g_Width / g_Height * frustrumTop;

	znear = 0.1f;
}

void drawGrid()
{
	glPushMatrix();

	glTranslatef(-(gridL / 2), 0, -(gridL / 2));

	glColor3f(.3, .3, .3);

	glBegin(GL_LINES);

	for (int i = 0; i <= gridL; i++)
	{
		glVertex3f(i, 0, 0);
		glVertex3f(i, 0, gridL);
		glVertex3f(0, 0, i);
		glVertex3f(gridL, 0, i);
	};

	glEnd();

	glPopMatrix();

	glPushMatrix();

	glTranslatef(-(gridL / 2), 0, -(gridL / 2));

	glColor3f(1, 0, 0);

	glBegin(GL_LINES);

	for (int i = 0; i <= gridL; i++)
	{
		glVertex3f(i, 0, 0);
		glVertex3f(i, gridL, 0);
		glVertex3f(0, i, 0);
		glVertex3f(gridL, i, 0);
	};

	glEnd();

	glPopMatrix();

	glPushMatrix();

	glTranslatef(-(gridL / 2), 0, -(gridL / 2));

	glColor3f(.3, .3, .3);

	glBegin(GL_LINES);

	for (int i = 0; i <= gridL; i++)
	{
		glVertex3f(0, i, 0);
		glVertex3f(0, i, gridL);
		glVertex3f(0, 0, i);
		glVertex3f(0, gridL, i);
	};

	glEnd();

	glPopMatrix();

	glPushMatrix();

	glTranslatef((gridL / 2), 0, -(gridL / 2));

	glColor3f(.3, .3, .3);

	glBegin(GL_LINES);

	for (int i = 0; i <= gridL; i++)
	{
		glVertex3f(0, i, 0);
		glVertex3f(0, i, gridL);
		glVertex3f(0, 0, i);
		glVertex3f(0, gridL, i);
	};

	glEnd();

	glPopMatrix();

	glPushMatrix();

	glTranslatef(-(gridL / 2), 0, (gridL / 2));

	glColor3f(.3, .3, .3);

	glBegin(GL_LINES);

	for (int i = 0; i <= gridL; i++)
	{
		glVertex3f(i, 0, 0);
		glVertex3f(i, gridL, 0);
		glVertex3f(0, i, 0);
		glVertex3f(gridL, i, 0);
	};

	glEnd();

	glPopMatrix();

	glPushMatrix();

	glTranslatef(-(gridL / 2), gridL, -(gridL / 2));

	glColor3f(.3, .3, .3);

	glBegin(GL_LINES);

	for (int i = 0; i <= gridL; i++)
	{
		glVertex3f(i, 0, 0);
		glVertex3f(i, 0, gridL);
		glVertex3f(0, 0, i);
		glVertex3f(gridL, 0, i);
	};

	glEnd();

	glPopMatrix();
}

void updateCV()
{
	cap >> frame;

	// loopzinho
	if (cap.get(CV_CAP_PROP_POS_FRAMES) == 500)
	{
		cap.set(CV_CAP_PROP_POS_FRAMES, 10);
	}

	imshow("video", frame);
}

// aqui o sistema de coordenadas da tela est� variando de -1 a 1 no eixo x e y
void mydisplay()
{
	// OpenCV Processing

	updateCV();

	// End of OpenCV Processing

	glMatrixMode(GL_PROJECTION);

	glLoadIdentity();

	glViewport(0, 0, g_Width, g_Height);

	glFrustum(frustrumLeft, frustrumRight, frustrumBottom, frustrumTop, znear, zfar);

	glMatrixMode(GL_MODELVIEW);

	glClearColor(0, 0, 0, 0);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	GLfloat extrinsic[16] =
	{
		Ext[0][0], Ext[1][0], Ext[2][0], Ext[3][0],
		Ext[0][1], Ext[1][1], Ext[2][1], Ext[3][1],
		Ext[0][2], Ext[1][2], Ext[2][2], Ext[3][2],
		Ext[0][3], Ext[1][3], Ext[2][3], Ext[3][3],
	};

	glMatrixMode(GL_MODELVIEW);

	glLoadMatrixf(extrinsic);

	glPushMatrix();

	glTranslatef(0, -(gridL / 2), 9);

	drawGrid();

	glPopMatrix();

	glColor3f(0.2, 0.0, 0.2);

	glPushMatrix();

	glTranslatef(0, 0, 9);
	glutWireSphere(0.3, 100, 100);

	glPopMatrix();

	glColor3f(0.2, 0.2, 0.0);

	glPushMatrix();

	glTranslatef(0.8, 0, 9);
	glutWireTeapot(0.3);

	glPopMatrix();

	glColor3f(0.0, 0.2, 0.2);

	glPushMatrix();

	glTranslatef(-0.6, 0, 9);
	glutWireCube(0.3);

	glPopMatrix();

	glFlush();
	glutPostRedisplay();
	glutSwapBuffers();
}

void handleKeyboardPressed(unsigned char key, int x, int y){
	buffer[(int) key] = true;
}

void handleKeyboardUp(unsigned char key, int x, int y){
	buffer[(int) key] = false;
}

void idleFunction()
{
	if (buffer['w'] == true & !buffer['b'])//camera pra frente 
		cameraTranslate(0, 0, translateCameraConst);
	if (buffer['W'] == true & !buffer['b'])//camera pra frente
		cameraTranslate(0, 0, translateCameraConst);
	if (buffer['s'] == true & !buffer['b'])
		cameraTranslate(0, 0, -translateCameraConst);
	if (buffer['S'] == true & !buffer['b'])
		cameraTranslate(0, 0, -translateCameraConst);
	if (buffer['a'] == true & !buffer['b'])
		cameraTranslate(translateCameraConst, 0, 0);
	if (buffer['A'] == true & !buffer['b'])
		cameraTranslate(translateCameraConst, 0, 0);
	if (buffer['d'] == true & !buffer['b'])
		cameraTranslate(-translateCameraConst, 0, 0);
	if (buffer['D'] == true & !buffer['b'])
		cameraTranslate(-translateCameraConst, 0, 0);
	if (buffer['j'] == true & !buffer['b'])
		cameraRotateY(rotateCameraConst);
	if (buffer['l'] == true & !buffer['b'])
		cameraRotateY(-rotateCameraConst);
	if (buffer['J'] == true & !buffer['b'])
		cameraRotateY(rotateCameraConst);
	if (buffer['L'] == true & !buffer['b'])
		cameraRotateY(-rotateCameraConst);
	if (buffer[27] == true)//ESC
		FimDoPrograma();
}

int main(int argc, char **argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
	glutInitWindowSize(800, 600);
	glutInitWindowPosition(0, 0);
	glutCreateWindow("OpenGL");
	glutDisplayFunc(mydisplay);
	glutReshapeFunc(myreshape);
	glutKeyboardUpFunc(handleKeyboardUp);
	glutKeyboardFunc(handleKeyboardPressed);
	glutIdleFunc(idleFunction);
	initialize();
	glutMainLoop();
	return 0;
}




