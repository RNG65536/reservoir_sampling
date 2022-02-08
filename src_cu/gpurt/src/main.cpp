#include <GL/glew.h>
//
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <omp.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <list>
#include <memory>
#include <string>
#include <vector>

#include "core/camera.h"
#include "core/common.h"
#include "core/cudapbo.h"
#include "core/rendertarget.h"
#include "io/obj_loader.h"
#include "renderer/lights.h"
#include "renderer/scene.h"

using std::cout;
using std::endl;

hvec<Vector3> view_image;
CudaPbo       g_pbo;

namespace megakernel_pt
{
void rendererInitialize(const TriangleMesh&              mesh,
                        int                              width,
                        int                              height,
                        const std::vector<MaterialSpec>& materials,
                        const Lights&                    lights);
void render_cuda(Vector3*         extern_image,
                 int              width,
                 int              height,
                 int              max_depth,
                 const glm::mat4& view_matrix,
                 int&             total_spp);

void setLaunchParams(const uint32_t tile_size[3], const uint32_t block_size[3]);

void setRenderer(int idx);
}  // namespace megakernel_pt

namespace path_tracer = megakernel_pt;

void gpuTonemap(Vector3*           accumbuf,
                const unsigned int framenumber,
                Vector3*           outputbuf,
                int                scrwidth,
                int                scrheight);

namespace Config
{
const int width  = 1280;
const int height = 720;
const int posx   = 100;
const int posy   = 100;

// must match screen size to use temporal filtering
const uint32_t tile_width  = width;
const uint32_t tile_height = height;
}  // namespace Config

std::unique_ptr<RenderRecord> rr;
GLuint                        g_renderTarget;

TriangleMesh              g_mesh;
std::vector<MaterialSpec> g_materials;
Lights                    g_lights;

void build_light_sampler(const TriangleMesh& scene, const std::vector<MaterialSpec>& materials)
{
    g_lights.build(scene, materials);
}

void initDefaultScene()
{
    glm::mat4 xform(1.0f);

    {
        MaterialSpec mat(Vector3(1, 0, 0), 1.0f, LGHT);
        g_materials.push_back(mat);
    }
    {
        MaterialSpec mat(Vector3(0, 1, 0), 1.0f, LGHT);
        g_materials.push_back(mat);
    }
    {
        MaterialSpec mat(Vector3(0, 0, 1), 1.0f, LGHT);
        g_materials.push_back(mat);
    }
    {
        MaterialSpec mat(Vector3(1, 1, 1), 1.0f, LGHT);
        g_materials.push_back(mat);
    }
    {
        MaterialSpec mat(Vector3(0, 1, 1), 1.0f, LGHT);
        g_materials.push_back(mat);
    }
    {
        MaterialSpec mat(Vector3(1, 0, 1), 1.0f, LGHT);
        g_materials.push_back(mat);
    }
    {
        MaterialSpec mat(Vector3(1, 1, 0), 1.0f, LGHT);
        g_materials.push_back(mat);
    }
    {
        Vector3      white(0.9, 0.9, 0.9);
        MaterialSpec mat(white, 0, DIFF);
        g_materials.push_back(mat);
    }

    //
    {
        xform = glm::mat4(1.0f);
        xform = glm::scale(glm::mat4(1.0f), glm::vec3(0.05f)) * xform;
        xform = glm::translate(glm::mat4(1.0f), glm::vec3(2.0, 1.0, 2.0)) * xform;

        {
            for (int x = 0; x < 5; x++)
            {
                for (int y = 0; y < 5; y++)
                {
                    xform = glm::mat4(1.0f);
                    xform = glm::scale(glm::mat4(1.0f), glm::vec3(16.0f)) * xform;
                    xform =
                        glm::translate(glm::mat4(1.0f), glm::vec3(x * 8.0, -3.0, y * 8.0)) * xform;

                    load_mesh(g_mesh, RandInt(0, 7), "../data/bunny_1k.obj", xform);  // LGHT
                }
            }
        }
    }

    {
        // floor
        float scale = 200.0f;
        load_quad(g_mesh,
                  7,
                  glm::translate(glm::mat4(1.0f), glm::vec3(0.0, -1.28, 0.0)) *
                      glm::scale(glm::mat4(1.0f), glm::vec3(scale)));
    }

    //
    g_mesh.buildBoundingBoxes();

    build_light_sampler(g_mesh, g_materials);

    path_tracer::rendererInitialize(
        g_mesh, Config::width, Config::height, g_materials, g_lights);
    view_image.resize(Config::width * Config::height);
    view_image.assign(view_image.size(), Vector3(0, 0, 0));
}

void raytracingInitialize()
{
    const uint32_t tile[3]  = {Config::tile_width, Config::tile_height, 1};
    const uint32_t block[3] = {8, 8, 1};
    path_tracer::setLaunchParams(tile, block);

    printf("building bvh...\n");

    initDefaultScene();
    printf("finished\n");

    g_pbo.init(Config::width, Config::height);
    rr->reset();
}

void raytracingFinalize()
{
    ;
}

SimpleCamera g_cam;

void raytracingUpdate()
{
    path_tracer::render_cuda(
        (Vector3*)rr->data(), Config::width, Config::height, -1, g_cam.viewMatrix(), rr->spp);

    Vector3* fb1 = (Vector3*)g_pbo.mapPbo(false);
    gpuTonemap((Vector3*)rr->data(), rr->spp, fb1, Config::width, Config::height);
    g_pbo.unmapPbo();
}

void display()
{
    glClear(GL_COLOR_BUFFER_BIT);

    raytracingUpdate();

    g_pbo.updateTexture();
    g_pbo.bindTexture();
    glBegin(GL_QUADS);
    glTexCoord2f(0, 1);
    glVertex2f(-1, -1);
    glTexCoord2f(1, 1);
    glVertex2f(1, -1);
    glTexCoord2f(1, 0);
    glVertex2f(1, 1);
    glTexCoord2f(0, 0);
    glVertex2f(-1, 1);
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);

    glutSwapBuffers();
    glutPostRedisplay();
}

int  ax, ay, mouse_states[3] = {GLUT_UP, GLUT_UP, GLUT_UP};
void mouse(int button, int state, int x, int y)
{
    mouse_states[button] = state;
    ax                   = x;
    ay                   = y;

    if (state == GLUT_UP)
    {
        g_cam.saveCameraParams();
    }
}

void motion(int x, int y)
{
    int dx = x - ax;
    int dy = y - ay;
    ax     = x;
    ay     = y;

    if (mouse_states[GLUT_LEFT_BUTTON] == GLUT_DOWN)
    {
        g_cam.rotate(dx, dy);
    }
    else if (mouse_states[GLUT_RIGHT_BUTTON] == GLUT_DOWN)
    {
        g_cam.pan(dx, dy);
    }
    else if (mouse_states[GLUT_MIDDLE_BUTTON] == GLUT_DOWN)
    {
        g_cam.zoom(dy);
    }

    rr->reset();
}

void scroll(int button, int dir, int x, int y)
{
    g_cam.zoom(-dir);

    rr->reset();
}

void keyboard(unsigned char key, int x, int y)
{
    static int i = 0;

    if (key == ' ')
    {
        i += 1;
        megakernel_pt::setRenderer(i);
    }

    rr->reset();
}

int main(int argc, char* argv[])
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(Config::width, Config::height);
    glutInitWindowPosition(Config::posx, Config::posy);
    glutCreateWindow("gpu path tracing");

    glewInit();
    if (!glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object"))
    {
        printf("Required OpenGL extensions missing.");
        exit(EXIT_FAILURE);
    }

    // testCudaTexture();

    rr = std::make_unique<RenderRecord>(Config::width, Config::height);

    raytracingInitialize();

    glGenTextures(1, &g_renderTarget);
    glBindTexture(GL_TEXTURE_2D, g_renderTarget);
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGB32F, Config::width, Config::height, 0, GL_RGB, GL_FLOAT, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

    glEnable(GL_TEXTURE_2D);
    glDisable(GL_DEPTH_TEST);
    glutReportErrors();

    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    glutMouseWheelFunc(scroll);

    glutMainLoop();

    return 0;
}
