#ifndef QDCS3DVIS_H
#define QDCS3DVIS_H

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLBuffer>


class QDCS3Dvis : public QOpenGLWidget, protected QOpenGLFunctions
{

public:
    explicit QDCS3Dvis(QWidget *parent = 0);
    ~QDCS3Dvis();
signals:

public slots:

protected:
    void initializeGL();
    void paintGL();
    void resizeGL(int width, int height);

    QSize minimumSizeHint() const;
    QSize sizeHint() const;
    void mousePressEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void wheelEvent(QWheelEvent *event);

    void setXRotation(int angle);
    void setYRotation(int angle);
    void setZRotation(int angle);
    void setUScale(float scale);
    void setXPan(float xpan);
    void setYPan(float ypan);

    bool loadOBJ(const char *path,
        std::vector <QVector3D> &out_vertices,
        std::vector <QVector2D> &out_uvs,
        std::vector <QVector3D> &out_normals
    );

private:
    void draw();

    int xRot;
    int yRot;
    int zRot;
    float xPan;
    float yPan;
    float uScale;

    QPoint lastPos;

    QOpenGLBuffer _vbo1_index;

    GLuint baseModelVertexBuffer;
    GLuint baseModelUVBuffer;
    GLuint baseModelTexture;

    QOpenGLShaderProgram *programShader;

    std::vector <QVector3D> baseVertices;
    std::vector <QVector2D> baseUVs;
    std::vector <QVector3D> baseNormals;

    static inline std::vector<std::string> split(std::string s, std::string delimiter)
    {
        size_t last = 0;
        size_t next = std::string::npos;

        std::vector<std::string> tokens;
        std::string token;

        while ((next = s.find(delimiter, last)) != std::string::npos)
        {
            token = s.substr(last, next - last);

            last = next + delimiter.length();

            tokens.push_back(token);
        }

        tokens.push_back(s.substr(last, next));

        return tokens;
    }
};

#endif // QDCS3DVIS_H
