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

    void setXRotation(int angle);
    void setYRotation(int angle);
    void setZRotation(int angle);

private:
    void draw();

    int xRot;
    int yRot;
    int zRot;

    QPoint lastPos;

    QOpenGLBuffer _vbo1_index;
};

#endif // QDCS3DVIS_H
