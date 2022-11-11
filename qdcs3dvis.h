#ifndef QDCS3DVIS_H
#define QDCS3DVIS_H

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLBuffer>
#include <QOpenGLTexture>

#include <QTimer>
#include <QDebug>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QWheelEvent>
#include <iostream>
#include <fstream>


#include "simuGlobals.hh"


class QDCS3Dvis : public QOpenGLWidget, protected QOpenGLFunctions
{

public:
    explicit QDCS3Dvis(QWidget *parent = 0);
    ~QDCS3Dvis();

    void setDelrot(float rot);
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
    void drawParallel(QMatrix4x4 &m);
    void drawAntiParallel(QMatrix4x4 &m);
    void drawParallelText(QMatrix4x4 &m);
    void drawAntiParallelText(QMatrix4x4 &m);
    void drawObject(std::vector<QVector3D> vertices, GLuint vbo, GLuint uvb);

    int xRot;
    int yRot;
    int zRot;
    float xPan;
    float yPan;
    float uScale;

    float x_first_crys = 0.4;
    float S_sour_y = 0.1;
    float text_scale = 4.0f;
    float text_voffset = 1.0f;
    float teta_crys1;
    float tetaref;

    float source_posx;
    float source_posy;
    float source_posz;

    float ap_posx;
    float ap_posy;
    float ap_posz;

    float c1_angle;

    float c2_angle_para;
    float c2_angle_anti;
    float c2_posx;
    float c2_posy;
    float c2_posz;

    float detec_angle_para;
    float detec_angle_anti;
    float detec_posx;
    float detec_posy;
    float detec_posz;

    float table_angle;
    float table_posx;
    float table_posy;
    float table_posz;

    float delrot;

    QPoint lastPos;

    QOpenGLBuffer _vbo1_index;

    GLuint baseCubeModelVertexBuffer;
    GLuint baseCubeModelUVBuffer;
    GLuint baseCubeModelTexture;
    QOpenGLTexture *baseCubeTexture;

    std::vector <QVector3D> baseCubeVertices;
    std::vector <QVector2D> baseCubeUVs;
    std::vector <QVector3D> baseCubeNormals;

    GLuint baseCylinderModelVertexBuffer;
    GLuint baseCylinderModelUVBuffer;
    GLuint baseCylinderModelTexture;
    QOpenGLTexture *baseCylinderTexture;

    std::vector <QVector3D> baseCylinderVertices;
    std::vector <QVector2D> baseCylinderUVs;
    std::vector <QVector3D> baseCylinderNormals;

    GLuint crystalCubeModelTexture;
    QOpenGLTexture *crystalCubeTexture;

    GLuint tableCubeModelTexture;
    QOpenGLTexture *tableCubeTexture;

    GLuint sourceCylinderModelTexture;
    QOpenGLTexture *sourceCylinderTexture;

    GLuint appertureCylinderModelTexture;
    QOpenGLTexture *appertureCylinderTexture;

    GLuint pillarCylinderModelTexture;
    QOpenGLTexture *pillarCylinderTexture;

    GLuint detecCylinderModelTexture;
    QOpenGLTexture *detecCylinderTexture;



    GLuint C1TextModelVertexBuffer;
    GLuint C1TextModelUVBuffer;
    GLuint C1TextModelTexture;
    QOpenGLTexture *C1TextTexture;

    std::vector <QVector3D> C1TextVertices;
    std::vector <QVector2D> C1TextUVs;
    std::vector <QVector3D> C1TextNormals;

    GLuint C2TextModelVertexBuffer;
    GLuint C2TextModelUVBuffer;
    GLuint C2TextModelTexture;
    QOpenGLTexture *C2TextTexture;

    std::vector <QVector3D> C2TextVertices;
    std::vector <QVector2D> C2TextUVs;
    std::vector <QVector3D> C2TextNormals;

    GLuint SourceTextModelVertexBuffer;
    GLuint SourceTextModelUVBuffer;
    GLuint SourceTextModelTexture;
    QOpenGLTexture *SourceTextTexture;

    std::vector <QVector3D> SourceTextVertices;
    std::vector <QVector2D> SourceTextUVs;
    std::vector <QVector3D> SourceTextNormals;

    GLuint AppertureTextModelVertexBuffer;
    GLuint AppertureTextModelUVBuffer;
    GLuint AppertureTextModelTexture;
    QOpenGLTexture *AppertureTextTexture;

    std::vector <QVector3D> AppertureTextVertices;
    std::vector <QVector2D> AppertureTextUVs;
    std::vector <QVector3D> AppertureTextNormals;

    GLuint TableTextModelVertexBuffer;
    GLuint TableTextModelUVBuffer;
    GLuint TableTextModelTexture;
    QOpenGLTexture *TableTextTexture;

    std::vector <QVector3D> TableTextVertices;
    std::vector <QVector2D> TableTextUVs;
    std::vector <QVector3D> TableTextNormals;

    GLuint DetectorTextModelVertexBuffer;
    GLuint DetectorTextModelUVBuffer;
    GLuint DetectorTextModelTexture;
    QOpenGLTexture *DetectorTextTexture;

    std::vector <QVector3D> DetectorTextVertices;
    std::vector <QVector2D> DetectorTextUVs;
    std::vector <QVector3D> DetectorTextNormals;

    GLuint ParaTextModelVertexBuffer;
    GLuint ParaTextModelUVBuffer;
    GLuint ParaTextModelTexture;
    QOpenGLTexture *ParaTextTexture;

    std::vector <QVector3D> ParaTextVertices;
    std::vector <QVector2D> ParaTextUVs;
    std::vector <QVector3D> ParaTextNormals;

    GLuint AntiTextModelVertexBuffer;
    GLuint AntiTextModelUVBuffer;
    GLuint AntiTextModelTexture;
    QOpenGLTexture *AntiTextTexture;

    std::vector <QVector3D> AntiTextVertices;
    std::vector <QVector2D> AntiTextUVs;
    std::vector <QVector3D> AntiTextNormals;


    QOpenGLShaderProgram *programShader;

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
