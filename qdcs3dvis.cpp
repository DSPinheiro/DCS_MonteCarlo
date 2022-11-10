#include "qdcs3dvis.h"

#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>

#include <QTimer>
#include <QDebug>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QWheelEvent>
#include <iostream>
#include <fstream>

extern char File_simu[200];

QDCS3Dvis::QDCS3Dvis(QWidget *parent)
    : _vbo1_index(QOpenGLBuffer::IndexBuffer)
{
    xRot = 0;
    yRot = 0;
    zRot = 0;
    uScale = 0.2;
    xPan = 0.0f;
    yPan = 0.0f;
}

QDCS3Dvis::~QDCS3Dvis()
{
    glDeleteBuffers(1, &baseModelVertexBuffer);
    glDeleteBuffers(1, &baseModelUVBuffer);
    glDeleteTextures(1, &baseModelTexture);
    glDeleteProgram(programShader->programId());
}

QSize QDCS3Dvis::minimumSizeHint() const
{
    return QSize(50, 50);
}

QSize QDCS3Dvis::sizeHint() const
{
    return QSize(400, 400);
}

static void qNormalizeAngle(int &angle)
{
    if (angle < 0)
        angle = 360;
    else if (angle > 360)
        angle = 0;
}

static void qNormalizeUScale(float &scale)
{
    if (scale < 0.01)
        scale = 0.01;
    else if (scale > 10)
        scale = 10;
}

void QDCS3Dvis::setXRotation(int angle)
{
    qNormalizeAngle(angle);
    if (angle != xRot) {
        xRot = angle;
        update();
    }
}

void QDCS3Dvis::setYRotation(int angle)
{
    qNormalizeAngle(angle);
    if (angle != yRot) {
        yRot = angle;
        update();
    }
}

void QDCS3Dvis::setZRotation(int angle)
{
    qNormalizeAngle(angle);
    if (angle != zRot) {
        zRot = angle;
        update();
    }
}

void QDCS3Dvis::setUScale(float scale)
{
    qNormalizeUScale(scale);
    if (scale != uScale) {
        uScale = scale;
        update();
    }
}

void QDCS3Dvis::setXPan(float xpan)
{
    if (xpan != xPan) {
        xPan = xpan;
        update();
    }
}

void QDCS3Dvis::setYPan(float ypan)
{
    if (ypan != yPan) {
        yPan = ypan;
        update();
    }
}

bool QDCS3Dvis::loadOBJ(const char *path,
    std::vector <QVector3D> &out_vertices,
    std::vector <QVector2D> &out_uvs,
    std::vector <QVector3D> &out_normals
)
{
    std::vector< unsigned int > vertexIndices, uvIndices, normalIndices;
    std::vector<QVector3D> temp_vertices;
    std::vector<QVector2D> temp_uvs;
    std::vector<QVector3D> temp_normals;

    std::ifstream objPath(path);
    if(!objPath.is_open()){
        std::cout << "Impossible to open the file at: " << path << std::endl;
        return false;
    }

    std::string lineHeader;

    while(getline(objPath, lineHeader)){
        // read the first word of the line
        if ( split(lineHeader, " " )[0] == "v" ){
            float vertex_x, vertex_y, vertex_z;
            sscanf(lineHeader.c_str(), "%*s %f %f %f\n", &vertex_x, &vertex_y, &vertex_z );

            QVector3D vertex;
            vertex.setX(vertex_x);
            vertex.setY(vertex_y);
            vertex.setZ(vertex_z);

            temp_vertices.push_back(vertex);
        }else if ( split(lineHeader, " " )[0] == "vt" ){
            float uv_x, uv_y;
            sscanf(lineHeader.c_str(), "%*s %f %f\n", &uv_x, &uv_y );

            QVector2D uv;
            uv.setX(uv_x);
            uv.setY(uv_y);

            temp_uvs.push_back(uv);
        }else if ( split(lineHeader, " " )[0] == "vn" ){
            float normal_x, normal_y, normal_z;
            sscanf(lineHeader.c_str(), "%*s %f %f %f\n", &normal_x, &normal_y, &normal_z );

            QVector3D normal;
            normal.setX(normal_x);
            normal.setY(normal_y);
            normal.setZ(normal_z);

            temp_normals.push_back(normal);
        }else if ( split(lineHeader, " " )[0] == "f" ){
            unsigned int vertexIndex[3], uvIndex[3], normalIndex[3];
            int matches = sscanf(lineHeader.c_str(), "%*s %d/%d/%d %d/%d/%d %d/%d/%d\n", &vertexIndex[0], &uvIndex[0], &normalIndex[0], &vertexIndex[1], &uvIndex[1], &normalIndex[1], &vertexIndex[2], &uvIndex[2], &normalIndex[2] );
            if (matches != 9){
                printf("File can't be read by our simple parser : ( Try exporting with other options\n");
                return false;
            }

            vertexIndices.push_back(vertexIndex[0]);
            vertexIndices.push_back(vertexIndex[1]);
            vertexIndices.push_back(vertexIndex[2]);
            uvIndices    .push_back(uvIndex[0]);
            uvIndices    .push_back(uvIndex[1]);
            uvIndices    .push_back(uvIndex[2]);
            normalIndices.push_back(normalIndex[0]);
            normalIndices.push_back(normalIndex[1]);
            normalIndices.push_back(normalIndex[2]);
        }
    }

    for( unsigned int i=0; i<vertexIndices.size(); i++ ){
        unsigned int vertexIndex = vertexIndices[i];
        QVector3D vertex = temp_vertices[ vertexIndex-1 ];
        out_vertices.push_back(vertex);
    }

    for( unsigned int i=0; i<uvIndices.size(); i++ ){
        unsigned int uvIndex = uvIndices[i];
        QVector2D uv = temp_uvs[ uvIndex-1 ];
        out_uvs.push_back(uv);
    }

    for( unsigned int i=0; i<normalIndices.size(); i++ ){
        unsigned int normalIndex = normalIndices[i];
        QVector3D normal = temp_normals[ normalIndex-1 ];
        out_normals.push_back(normal);
    }

    return true;
}

void QDCS3Dvis::initializeGL()
{
    initializeOpenGLFunctions();

    glClearColor(0.2f, 0.2f, 0.2f, 1.0f);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glShadeModel(GL_SMOOTH);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);

    static GLfloat lightPosition[4] = { 0, 0, 10, 1.0 };
    glLightfv(GL_LIGHT0, GL_POSITION, lightPosition);

    QOpenGLShader *vshader = new QOpenGLShader(QOpenGLShader::Vertex, this);
    const char *vsrc =
        "attribute highp vec4 vertex;\n\
        attribute mediump vec4 texCoord;\n\
        varying mediump vec4 texc;\n\
        uniform mediump mat4 matrix;\n\
        void main(void)\n\
        {\n\
            gl_Position = matrix * vertex;\n\
            texc = texCoord;\n\
        }\n";
    vshader->compileSourceCode(vsrc);

    QOpenGLShader *fshader = new QOpenGLShader(QOpenGLShader::Fragment, this);
    const char *fsrc =
        "uniform sampler2D texture;\n\
        varying mediump vec4 texc;\n\
        void main(void)\n\
        {\n\
            gl_FragColor = texture2D(texture, texc.st);\n\
        }\n";
    fshader->compileSourceCode(fsrc);

    programShader = new QOpenGLShaderProgram;
    programShader->addShader(vshader);
    programShader->addShader(fshader);
    programShader->bindAttributeLocation("vertex", 0);
    programShader->bindAttributeLocation("texCoord", 1);

    programShader->link();
    programShader->bind();

    programShader->setUniformValue("texture", 0);

    //Load base model from disk into arrays
    std::string baseModelPath = std::string(File_simu) + "\\DCSModels\\cube.obj";

    bool res = loadOBJ(baseModelPath.c_str(), baseVertices, baseUVs, baseNormals);

    if (res)
        std::cout << "Base model loaded successfully." << std::endl;
    else
        std::cout << "Error loading base model." << std::endl;


    std::string baseTexturePath = std::string(File_simu) + "\\DCSModels\\cubeTex.png";

    QOpenGLTexture *baseTexture = new QOpenGLTexture(QImage(baseTexturePath.c_str()).mirrored());

    //Load the base model into OpenGL buffer objects
    glGenBuffers(1, &baseModelVertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, baseModelVertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, baseVertices.size() * sizeof(QVector3D), &baseVertices[0], GL_STATIC_DRAW);

    glGenBuffers(1, &baseModelUVBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, baseModelUVBuffer);
    glBufferData(GL_ARRAY_BUFFER, baseUVs.size() * sizeof(QVector2D), &baseUVs[0], GL_STATIC_DRAW);

    baseTexture->bind();
    baseModelTexture = baseTexture->textureId();
}

void QDCS3Dvis::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    QMatrix4x4 m;
    m.ortho(-0.5f, 0.5f, 0.5f, -0.5f, 4.0f, 15.0f);
    m.translate(xPan, yPan, -10.0);
    m.scale(uScale);
    m.rotate(xRot, 1.0, 0.0, 0.0);
    m.rotate(yRot, 0.0, 1.0, 0.0);
    m.rotate(zRot, 0.0, 0.0, 1.0);

    programShader->setUniformValue("matrix", m);
    programShader->enableAttributeArray(0);
    programShader->enableAttributeArray(1);
    programShader->setAttributeBuffer(0, GL_FLOAT, 0, 3, 5 * sizeof(GLfloat));
    programShader->setAttributeBuffer(1, GL_FLOAT, 3 * sizeof(GL_FLOAT), 2, 5 * sizeof(GLfloat));

    draw();
}

void QDCS3Dvis::resizeGL(int width, int height)
{
    int side = qMin(width, height);
    glViewport((width - side) / 2, (height - side) / 2, side, side);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
#ifdef QT_OPENGL_ES_1
    glOrthof(-2, +2, -2, +2, 1.0, 15.0);
#else
    glOrtho(-2, +2, -2, +2, 1.0, 15.0);
#endif
    glMatrixMode(GL_MODELVIEW);
}

void QDCS3Dvis::mousePressEvent(QMouseEvent *event)
{
    lastPos = event->pos();
}

void QDCS3Dvis::mouseMoveEvent(QMouseEvent *event)
{
    int dx = event->x() - lastPos.x();
    int dy = event->y() - lastPos.y();

    if (event->buttons() & Qt::LeftButton) {
        setXRotation(xRot + dy);
        setYRotation(yRot + dx);
    } else if (event->buttons() & Qt::RightButton) {
        setXRotation(xRot + dy);
        setZRotation(zRot + dx);
    } else if (event->buttons() & Qt::MiddleButton) {
        setXPan(xPan + dx / 200.0);
        setYPan(yPan + dy / 200.0);
    }

    lastPos = event->pos();
}

void QDCS3Dvis::wheelEvent(QWheelEvent *event)
{
    setUScale(uScale + event->angleDelta().y() / 12000.0);
}

void QDCS3Dvis::draw()
{
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, baseModelTexture);

    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, baseModelVertexBuffer);
    glVertexAttribPointer(
        0,                  // attribute
        3,                  // size
        GL_FLOAT,           // type
        GL_FALSE,           // normalized?
        0,                  // stride
        (void*)0            // array buffer offset
    );


    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, baseModelUVBuffer);
    glVertexAttribPointer(
        1,                                // attribute
        2,                                // size
        GL_FLOAT,                         // type
        GL_FALSE,                         // normalized?
        0,                                // stride
        (void*)0                          // array buffer offset
    );


    glDrawArrays(GL_TRIANGLES, 0, baseVertices.size() );

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
}
