#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

struct Camera
{
    glm::vec2 resolution;
    glm::vec3 position;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec2 fov;
    float     apertureRadius;
    float     focalDistance;
};

// new camera
class SimpleCamera
{
public:
    glm::vec3 m_origin;
    glm::vec3 m_lookat;
    glm::vec3 m_up;
    glm::vec3 m_direction;
    glm::vec3 m_frame_x;
    glm::vec3 m_frame_y;

    SimpleCamera(
        //         const glm::vec3& origin = glm::vec3(0.0f, 3.0f, 5.0f),
        //         const glm::vec3& lookat = glm::vec3(0.0f, 0.0f, 0.0f),
        //         const glm::vec3& up = glm::vec3(0.0f, 1.0f, 0.0f))
        const glm::vec3& origin = glm::vec3(3.4555792808532715,
                                            1.2124358415603638,
                                            3.2989654541015625),
        const glm::vec3& lookat = glm::vec3(0.0942695364356041,
                                            1.1369876861572266,
                                            0.39623117446899414),
        const glm::vec3& up     = glm::vec3(0.0f, 1.0f, 0.0f))
    {
        if (!loadCameraParams())
        {
            m_origin = origin;
            m_lookat = lookat;
            m_up     = normalize(up);
        }

        m_direction = normalize(m_lookat - m_origin);
        m_frame_x   = normalize(cross(m_direction, m_up));
        m_frame_y   = normalize(cross(m_frame_x, m_direction));
    }

    ~SimpleCamera() { saveCameraParams(); }

    bool loadCameraParams()
    {
        const char*   filename = CAMERA_FILE;
        std::ifstream ifs(filename, std::ios::in | std::ios::binary);
        if (!ifs.is_open())
        {
            return false;
        }
        ifs >> m_origin.x >> m_origin.y >> m_origin.z;
        ifs >> m_lookat.x >> m_lookat.y >> m_lookat.z;
        ifs >> m_up.x >> m_up.y >> m_up.z;
        ifs.close();
        return true;
    }

    bool saveCameraParams() const
    {
        const char*   filename = CAMERA_FILE;
        std::ofstream ofs(filename, std::ios::out | std::ios::binary);
        if (!ofs.is_open())
        {
            return false;
        }
        ofs << m_origin.x << " " << m_origin.y << " " << m_origin.z << " ";
        ofs << m_lookat.x << " " << m_lookat.y << " " << m_lookat.z << " ";
        ofs << m_up.x << " " << m_up.y << " " << m_up.z << " ";
        ofs.close();
        return true;
    }

    glm::mat4 viewMatrix() const { return glm::lookAt(m_origin, m_lookat, m_up); }

    void pan(float x, float y)
    {
        m_lookat += m_frame_x * x * -0.01f;
        m_lookat += m_frame_y * y * 0.01f;
        m_origin += m_frame_x * x * -0.01f;
        m_origin += m_frame_y * y * 0.01f;
    }

    void rotate(float x, float y)
    {
        m_direction = m_lookat - m_origin;
        float dist  = glm::length(m_direction);
        m_direction *= 1.0f / dist;
        m_frame_x = normalize(cross(m_direction, m_up));
        m_frame_y = normalize(cross(m_frame_x, m_direction));
        m_up      = m_frame_y;

        m_origin += m_frame_x * (x * dist * -0.004f);
        m_origin += m_frame_y * (y * dist * 0.004f);
        m_origin = m_lookat + glm::normalize(m_origin - m_lookat) * dist;

        m_direction = normalize(m_lookat - m_origin);
        m_frame_x   = normalize(cross(m_direction, m_up));
        m_frame_y   = normalize(cross(m_frame_x, m_direction));
    }

    void zoom(float z)
    {
        // m_origin += m_direction * z * -0.2f;

        glm::vec3 diff  = m_origin - m_lookat;
        float     dist  = glm::length(diff);
        float     dist1 = fmaxf(dist * (1.0f + z * 0.01f), 0.01f);
        m_origin        = m_lookat + diff * (dist1 / dist);
    }

    float _width;
    float _height;
    float _fovx;
    float _fovy;

    void set_resolution(const int width, const int height)
    {
        _width  = width;
        _height = height;
        set_fov_x(_fovx);
    }

    void set_fov_x(float fov)
    {
        _fovx = fov;
        _fovy = glm::degrees(atan(tan(glm::radians(_fovx) * 0.5) * (_height / _width)) * 2.0);
    }

    void build_render_camera(Camera* render_camera)
    {
        render_camera->position       = m_origin;
        render_camera->view           = m_direction;
        render_camera->up             = m_up;
        render_camera->resolution     = glm::vec2(_width, _height);
        render_camera->fov            = glm::vec2(_fovx, _fovy);
        render_camera->apertureRadius = 0.0f;
        render_camera->focalDistance  = 4.0f;

#if 0
        printf(">> position = %f, %f, %f\n",
               render_camera->position.x,
               render_camera->position.y,
               render_camera->position.z);
        printf(">> direction = %f, %f, %f\n",
               render_camera->view.x,
               render_camera->view.y,
               render_camera->view.z);
        printf(">> up = %f, %f, %f\n",
               render_camera->up.x,
               render_camera->up.y,
               render_camera->up.z);
        printf(">> resolution = %f, %f\n",
               render_camera->resolution.x,
               render_camera->resolution.y);
        printf(">> fov = %f, %f\n", render_camera->fov.x, render_camera->fov.y);
        printf(">> aperture radius = %f\n", render_camera->apertureRadius);
        printf(">> focal distance = %f\n", render_camera->focalDistance);
#endif
    }
};
