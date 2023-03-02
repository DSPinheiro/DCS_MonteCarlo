#include "qdcs3dvis.h"



QDCS3Dvis::QDCS3Dvis(QWidget* parent)
    : _vbo1_index(QOpenGLBuffer::IndexBuffer)
{
    xRot = 0;
    yRot = 0;
    zRot = 0;
    uScale = 0.2f;
    xPan = 0.0f;
    yPan = 0.0f;

    delrot = 0.0f;
    tetaref = 90 - teta_crys1;
}

QDCS3Dvis::~QDCS3Dvis()
{
    glDeleteBuffers(1, &baseCubeModelVertexBuffer);
    glDeleteBuffers(1, &baseCubeModelUVBuffer);
    glDeleteTextures(1, &baseCubeModelTexture);

    glDeleteBuffers(1, &baseCylinderModelVertexBuffer);
    glDeleteBuffers(1, &baseCylinderModelUVBuffer);
    glDeleteTextures(1, &baseCylinderModelTexture);

    glDeleteTextures(1, &crystalCubeModelTexture);
    glDeleteTextures(1, &tableCubeModelTexture);

    glDeleteTextures(1, &sourceCylinderModelTexture);
    glDeleteTextures(1, &appertureCylinderModelTexture);
    glDeleteTextures(1, &pillarCylinderModelTexture);
    glDeleteTextures(1, &detecCylinderModelTexture);

    glDeleteTextures(1, &C1TextModelTexture);
    glDeleteTextures(1, &C2TextModelTexture);
    glDeleteTextures(1, &SourceTextModelTexture);
    glDeleteTextures(1, &AppertureTextModelTexture);
    glDeleteTextures(1, &TableTextModelTexture);
    glDeleteTextures(1, &DetectorTextModelTexture);
    glDeleteTextures(1, &ParaTextModelTexture);
    glDeleteTextures(1, &AntiTextModelTexture);

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

static void qNormalizeAngle(int& angle)
{
    if (angle < 0)
        angle = 360;
    else if (angle > 360)
        angle = 0;
}

static void qNormalizeUScale(float& scale)
{
    if (scale < 0.01f)
        scale = 0.01f;
    else if (scale > 50.0f)
        scale = 50.0f;
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

void QDCS3Dvis::setDelrot(float rot)
{
    delrot = rot;
}

void QDCS3Dvis::setEventsToTrace(std::vector<std::vector<double>> events_para, std::vector<std::vector<double>> events_anti)
{
    eventsToTrace_para = events_para;
    eventsToTrace_anti = events_anti;
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

    //Load base cube model from disk into arrays
    std::string baseCubeModelPath = std::string(File_simu) + "\\DCSModels\\cube.obj";

    bool res = loadOBJ(baseCubeModelPath.c_str(), baseCubeVertices, baseCubeUVs, baseCubeNormals);

    if (res)
        std::cout << "Base cube model loaded successfully." << std::endl;
    else
        std::cout << "Error loading base cube model." << std::endl;

    //Load base cube model from disk into arrays
    std::string baseCylinderModelPath = std::string(File_simu) + "\\DCSModels\\cylinder.obj";

    res = loadOBJ(baseCylinderModelPath.c_str(), baseCylinderVertices, baseCylinderUVs, baseCylinderNormals);

    if (res)
        std::cout << "Base cylinder model loaded successfully." << std::endl;
    else
        std::cout << "Error loading base cylinder model." << std::endl;

    //Load C1 text model from disk into arrays
    std::string C1TextModelPath = std::string(File_simu) + "\\DCSModels\\C1text.obj";

    res = loadOBJ(C1TextModelPath.c_str(), C1TextVertices, C1TextUVs, C1TextNormals);

    if (res)
        std::cout << "C1 Text model loaded successfully." << std::endl;
    else
        std::cout << "Error loading C1 Text model." << std::endl;

    //Load C2 text model from disk into arrays
    std::string C2TextModelPath = std::string(File_simu) + "\\DCSModels\\C2text.obj";

    res = loadOBJ(C2TextModelPath.c_str(), C2TextVertices, C2TextUVs, C2TextNormals);

    if (res)
        std::cout << "C2 Text model loaded successfully." << std::endl;
    else
        std::cout << "Error loading C2 Text model." << std::endl;

    //Load Source text model from disk into arrays
    std::string SourceTextModelPath = std::string(File_simu) + "\\DCSModels\\Sourcetext.obj";

    res = loadOBJ(SourceTextModelPath.c_str(), SourceTextVertices, SourceTextUVs, SourceTextNormals);

    if (res)
        std::cout << "Source Text model loaded successfully." << std::endl;
    else
        std::cout << "Error loading Source Text model." << std::endl;

    //Load Apperture text model from disk into arrays
    std::string AppertureTextModelPath = std::string(File_simu) + "\\DCSModels\\Apperturetext.obj";

    res = loadOBJ(AppertureTextModelPath.c_str(), AppertureTextVertices, AppertureTextUVs, AppertureTextNormals);

    if (res)
        std::cout << "Apperture Text model loaded successfully." << std::endl;
    else
        std::cout << "Error loading Apperture Text model." << std::endl;

    //Load Table text model from disk into arrays
    std::string TableTextModelPath = std::string(File_simu) + "\\DCSModels\\Tabletext.obj";

    res = loadOBJ(TableTextModelPath.c_str(), TableTextVertices, TableTextUVs, TableTextNormals);

    if (res)
        std::cout << "Table Text model loaded successfully." << std::endl;
    else
        std::cout << "Error loading Table Text model." << std::endl;

    //Load Detector text model from disk into arrays
    std::string DetectorTextModelPath = std::string(File_simu) + "\\DCSModels\\Detectortext.obj";

    res = loadOBJ(DetectorTextModelPath.c_str(), DetectorTextVertices, DetectorTextUVs, DetectorTextNormals);

    if (res)
        std::cout << "Detector Text model loaded successfully." << std::endl;
    else
        std::cout << "Error loading Detector Text model." << std::endl;

    //Load Para Configuration text model from disk into arrays
    std::string ParaTextModelPath = std::string(File_simu) + "\\DCSModels\\ParaConfigtext.obj";

    res = loadOBJ(ParaTextModelPath.c_str(), ParaTextVertices, ParaTextUVs, ParaTextNormals);

    if (res)
        std::cout << "Para Configuration Text model loaded successfully." << std::endl;
    else
        std::cout << "Error loading Para Configuration Text model." << std::endl;

    //Load Anti Configuration text model from disk into arrays
    std::string AntiTextModelPath = std::string(File_simu) + "\\DCSModels\\AntiConfigtext.obj";

    res = loadOBJ(AntiTextModelPath.c_str(), AntiTextVertices, AntiTextUVs, AntiTextNormals);

    if (res)
        std::cout << "Para Configuration Text model loaded successfully." << std::endl;
    else
        std::cout << "Error loading Para Configuration Text model." << std::endl;

    //Load base cube texture from disk
    std::string baseCubeTexturePath = std::string(File_simu) + "\\DCSModels\\cubeTex.png";

    baseCubeTexture = new QOpenGLTexture(QImage(baseCubeTexturePath.c_str()).mirrored());

    //Load crystal cube texture from disk
    std::string crystalCubeTexturePath = std::string(File_simu) + "\\DCSModels\\crystalTex.png";

    crystalCubeTexture = new QOpenGLTexture(QImage(crystalCubeTexturePath.c_str()).mirrored());

    //Load table cube texture from disk
    std::string tableCubeTexturePath = std::string(File_simu) + "\\DCSModels\\tableTex.png";

    tableCubeTexture = new QOpenGLTexture(QImage(tableCubeTexturePath.c_str()).mirrored());

    //Load base cylinder texture from disk
    std::string baseCylinderTexturePath = std::string(File_simu) + "\\DCSModels\\cylinderTex.png";

    baseCylinderTexture = new QOpenGLTexture(QImage(baseCylinderTexturePath.c_str()).mirrored());

    //Load source cylinder texture from disk
    std::string sourceCylinderTexturePath = std::string(File_simu) + "\\DCSModels\\sourceTex.png";

    sourceCylinderTexture = new QOpenGLTexture(QImage(sourceCylinderTexturePath.c_str()).mirrored());

    //Load apperture cylinder texture from disk
    std::string appertureCylinderTexturePath = std::string(File_simu) + "\\DCSModels\\apperTex.png";

    appertureCylinderTexture = new QOpenGLTexture(QImage(appertureCylinderTexturePath.c_str()).mirrored());

    //Load pillar cylinder texture from disk
    std::string pillarCylinderTexturePath = std::string(File_simu) + "\\DCSModels\\steelTex.png";

    pillarCylinderTexture = new QOpenGLTexture(QImage(pillarCylinderTexturePath.c_str()).mirrored());

    //Load detector cylinder texture from disk
    std::string detecCylinderTexturePath = std::string(File_simu) + "\\DCSModels\\detecTex.png";

    detecCylinderTexture = new QOpenGLTexture(QImage(detecCylinderTexturePath.c_str()).mirrored());

    //Load C1 text texture from disk
    std::string C1TextTexturePath = std::string(File_simu) + "\\DCSModels\\C1Tex.png";

    C1TextTexture = new QOpenGLTexture(QImage(C1TextTexturePath.c_str()).mirrored());

    //Load C2 text texture from disk
    std::string C2TextTexturePath = std::string(File_simu) + "\\DCSModels\\C2Tex.png";

    C2TextTexture = new QOpenGLTexture(QImage(C2TextTexturePath.c_str()).mirrored());

    //Load Source text texture from disk
    std::string SourceTextTexturePath = std::string(File_simu) + "\\DCSModels\\SourceTex.png";

    SourceTextTexture = new QOpenGLTexture(QImage(SourceTextTexturePath.c_str()).mirrored());

    //Load Apperture text texture from disk
    std::string AppertureTextTexturePath = std::string(File_simu) + "\\DCSModels\\AppertureTex.png";

    AppertureTextTexture = new QOpenGLTexture(QImage(AppertureTextTexturePath.c_str()).mirrored());

    //Load Table text texture from disk
    std::string TableTextTexturePath = std::string(File_simu) + "\\DCSModels\\TableTextTex.png";

    TableTextTexture = new QOpenGLTexture(QImage(TableTextTexturePath.c_str()).mirrored());

    //Load Detector text texture from disk
    std::string DetectorTextTexturePath = std::string(File_simu) + "\\DCSModels\\DetectorTex.png";

    DetectorTextTexture = new QOpenGLTexture(QImage(DetectorTextTexturePath.c_str()).mirrored());

    //Load Para Configuration texture from disk
    std::string ParaTextTexturePath = std::string(File_simu) + "\\DCSModels\\ParaConfigTex.png";

    ParaTextTexture = new QOpenGLTexture(QImage(ParaTextTexturePath.c_str()).mirrored());

    //Load Anti Configuration texture from disk
    std::string AntiTextTexturePath = std::string(File_simu) + "\\DCSModels\\AntiConfigTex.png";

    AntiTextTexture = new QOpenGLTexture(QImage(AntiTextTexturePath.c_str()).mirrored());

    //Load the base cube model into OpenGL buffer objects
    glGenBuffers(1, &baseCubeModelVertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, baseCubeModelVertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, baseCubeVertices.size() * sizeof(QVector3D), &baseCubeVertices[0], GL_STATIC_DRAW);

    glGenBuffers(1, &baseCubeModelUVBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, baseCubeModelUVBuffer);
    glBufferData(GL_ARRAY_BUFFER, baseCubeUVs.size() * sizeof(QVector2D), &baseCubeUVs[0], GL_STATIC_DRAW);

    //Load the base cylinder model into OpenGL buffer objects
    glGenBuffers(1, &baseCylinderModelVertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, baseCylinderModelVertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, baseCylinderVertices.size() * sizeof(QVector3D), &baseCylinderVertices[0], GL_STATIC_DRAW);

    glGenBuffers(1, &baseCylinderModelUVBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, baseCylinderModelUVBuffer);
    glBufferData(GL_ARRAY_BUFFER, baseCylinderUVs.size() * sizeof(QVector2D), &baseCylinderUVs[0], GL_STATIC_DRAW);

    //Load the C1 text model into OpenGL buffer objects
    glGenBuffers(1, &C1TextModelVertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, C1TextModelVertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, C1TextVertices.size() * sizeof(QVector3D), &C1TextVertices[0], GL_STATIC_DRAW);

    glGenBuffers(1, &C1TextModelUVBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, C1TextModelUVBuffer);
    glBufferData(GL_ARRAY_BUFFER, C1TextUVs.size() * sizeof(QVector2D), &C1TextUVs[0], GL_STATIC_DRAW);

    //Load the C2 text model into OpenGL buffer objects
    glGenBuffers(1, &C2TextModelVertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, C2TextModelVertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, C2TextVertices.size() * sizeof(QVector3D), &C2TextVertices[0], GL_STATIC_DRAW);

    glGenBuffers(1, &C2TextModelUVBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, C2TextModelUVBuffer);
    glBufferData(GL_ARRAY_BUFFER, C2TextUVs.size() * sizeof(QVector2D), &C2TextUVs[0], GL_STATIC_DRAW);

    //Load the Source text model into OpenGL buffer objects
    glGenBuffers(1, &SourceTextModelVertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, SourceTextModelVertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, SourceTextVertices.size() * sizeof(QVector3D), &SourceTextVertices[0], GL_STATIC_DRAW);

    glGenBuffers(1, &SourceTextModelUVBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, SourceTextModelUVBuffer);
    glBufferData(GL_ARRAY_BUFFER, SourceTextUVs.size() * sizeof(QVector2D), &SourceTextUVs[0], GL_STATIC_DRAW);

    //Load the Apperture text model into OpenGL buffer objects
    glGenBuffers(1, &AppertureTextModelVertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, AppertureTextModelVertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, AppertureTextVertices.size() * sizeof(QVector3D), &AppertureTextVertices[0], GL_STATIC_DRAW);

    glGenBuffers(1, &AppertureTextModelUVBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, AppertureTextModelUVBuffer);
    glBufferData(GL_ARRAY_BUFFER, AppertureTextUVs.size() * sizeof(QVector2D), &AppertureTextUVs[0], GL_STATIC_DRAW);

    //Load the Table text model into OpenGL buffer objects
    glGenBuffers(1, &TableTextModelVertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, TableTextModelVertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, TableTextVertices.size() * sizeof(QVector3D), &TableTextVertices[0], GL_STATIC_DRAW);

    glGenBuffers(1, &TableTextModelUVBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, TableTextModelUVBuffer);
    glBufferData(GL_ARRAY_BUFFER, TableTextUVs.size() * sizeof(QVector2D), &TableTextUVs[0], GL_STATIC_DRAW);

    //Load the Detector text model into OpenGL buffer objects
    glGenBuffers(1, &DetectorTextModelVertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, DetectorTextModelVertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, DetectorTextVertices.size() * sizeof(QVector3D), &DetectorTextVertices[0], GL_STATIC_DRAW);

    glGenBuffers(1, &DetectorTextModelUVBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, DetectorTextModelUVBuffer);
    glBufferData(GL_ARRAY_BUFFER, DetectorTextUVs.size() * sizeof(QVector2D), &DetectorTextUVs[0], GL_STATIC_DRAW);

    //Load the Para Configuration text model into OpenGL buffer objects
    glGenBuffers(1, &ParaTextModelVertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, ParaTextModelVertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, ParaTextVertices.size() * sizeof(QVector3D), &ParaTextVertices[0], GL_STATIC_DRAW);

    glGenBuffers(1, &ParaTextModelUVBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, ParaTextModelUVBuffer);
    glBufferData(GL_ARRAY_BUFFER, ParaTextUVs.size() * sizeof(QVector2D), &ParaTextUVs[0], GL_STATIC_DRAW);

    //Load the Anti Configuration text model into OpenGL buffer objects
    glGenBuffers(1, &AntiTextModelVertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, AntiTextModelVertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, AntiTextVertices.size() * sizeof(QVector3D), &AntiTextVertices[0], GL_STATIC_DRAW);

    glGenBuffers(1, &AntiTextModelUVBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, AntiTextModelUVBuffer);
    glBufferData(GL_ARRAY_BUFFER, AntiTextUVs.size() * sizeof(QVector2D), &AntiTextUVs[0], GL_STATIC_DRAW);



    baseCubeModelTexture = baseCubeTexture->textureId();
    crystalCubeModelTexture = crystalCubeTexture->textureId();
    tableCubeModelTexture = tableCubeTexture->textureId();
    baseCylinderModelTexture = baseCylinderTexture->textureId();
    sourceCylinderModelTexture = sourceCylinderTexture->textureId();
    appertureCylinderModelTexture = appertureCylinderTexture->textureId();
    pillarCylinderModelTexture = pillarCylinderTexture->textureId();
    detecCylinderModelTexture = detecCylinderTexture->textureId();

    C1TextModelTexture = C1TextTexture->textureId();
    C2TextModelTexture = C2TextTexture->textureId();
    SourceTextModelTexture = SourceTextTexture->textureId();
    AppertureTextModelTexture = AppertureTextTexture->textureId();
    TableTextModelTexture = TableTextTexture->textureId();
    DetectorTextModelTexture = DetectorTextTexture->textureId();
    ParaTextModelTexture = ParaTextTexture->textureId();
    AntiTextModelTexture = AntiTextTexture->textureId();



    source_posx = 0.0f;
    source_posy = - S_sour_y / 2 - GeoParapathlengthsInput.LT_aper - GeoParapathlengthsInput.dist_T_Cr1 - GeolengthelementsInput.y_first_crys / 2;
    source_posz = 0.0f;

    ap_posx = 0.0f;
    ap_posy = -GeoParapathlengthsInput.LT_aper / 2 - GeoParapathlengthsInput.dist_T_Cr1 - GeolengthelementsInput.y_first_crys / 2;
    ap_posz = 0.0f;

    c2_posx = GeoParapathlengthsInput.dist_Cr1_Cr2 + x_first_crys;
    c2_posy = 0.0f;
    c2_posz = 0.0f;

    detec_posx = GeoParapathlengthsInput.dist_Cr2_Det + GeolengthelementsInput.zdetc / 2;
    detec_posy = 0.0f;
    detec_posz = GeolengthelementsInput.shift_det_ver;

    table_posx = 1.5 * (GeoParapathlengthsInput.dist_Cr1_Cr2 + GeoParapathlengthsInput.dist_Cr2_Det) / 3;
    table_posy = 0.0f;
    table_posz = -15.0f;
}

void QDCS3Dvis::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    QMatrix4x4 m;
    m.ortho(-0.5f, 0.5f, 0.5f, -0.5f, 50.0f, -50.0f);
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

    drawParallel(m);
    drawAntiParallel(m);
    drawParallelText(m);
    drawAntiParallelText(m);

    drawParallelEvents(m);
    drawAntiparallelEvents(m);
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
    glOrtho(-0.5, +0.5, -0.5, +0.5, 4.0, 15.0);
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

void QDCS3Dvis::drawParallel(QMatrix4x4 &m)
{

    //Source Representation (i.e. entry flange to the chamber)
    m.translate(source_posx, source_posy, source_posz);
    m.scale((GeolengthelementsInput.S_sour + 1) / 2, S_sour_y / 2, (GeolengthelementsInput.S_sour + 1) / 2);

    programShader->setUniformValue("matrix", m);
    sourceCylinderTexture->bind();
    drawObject(baseCylinderVertices, baseCylinderModelVertexBuffer, baseCylinderModelUVBuffer);

    m.scale(2 / (GeolengthelementsInput.S_sour + 1), 2 / S_sour_y, 2 / (GeolengthelementsInput.S_sour + 1));
    m.translate(-source_posx, -source_posy, -source_posz);


    //Apperture Representation (Originally a Cu tube)
    m.translate(ap_posx, ap_posy, ap_posz);
    m.scale(GeolengthelementsInput.S_aper / 2, GeoParapathlengthsInput.LT_aper / 2, GeolengthelementsInput.S_aper / 2);

    programShader->setUniformValue("matrix", m);
    appertureCylinderTexture->bind();
    drawObject(baseCylinderVertices, baseCylinderModelVertexBuffer, baseCylinderModelUVBuffer);

    m.scale(2 / GeolengthelementsInput.S_aper, 2 / GeoParapathlengthsInput.LT_aper, 2 / GeolengthelementsInput.S_aper);
    m.translate(-ap_posx, -ap_posy, -ap_posz);
    

    table_angle = GeoParametersInput.teta_table + 90;
    c1_angle = GeoParametersInput.Exp_crys1 - GeoParametersInput.OffsetRotCry1;
    c2_angle_para = - table_angle - c1_angle + delrot * convdeg + 90;
    detec_angle_para = GeoParametersInput.teta_detec_para;

    
    //Crystal 1
    m.rotate(table_angle + c1_angle, 0.0, 0.0, 1.0);
    m.rotate(GeoParametersInput.tilt_C1, 0.0f, 1.0f, 0.0f);
    m.scale(x_first_crys / 2, GeolengthelementsInput.y_first_crys / 2, GeolengthelementsInput.z_first_crys / 2);

    programShader->setUniformValue("matrix", m);
    crystalCubeTexture->bind();
    drawObject(baseCubeVertices, baseCubeModelVertexBuffer, baseCubeModelUVBuffer);

    m.scale(2 / x_first_crys, 2 / GeolengthelementsInput.y_first_crys, 2 / GeolengthelementsInput.z_first_crys);
    m.rotate(-GeoParametersInput.tilt_C1, 0.0f, 1.0f, 0.0f);
    m.rotate(-table_angle - c1_angle, 0.0, 0.0, 1.0);

    //Crystal 1 Pillar
    m.translate(0.0f, 0.0f, table_posz / 2 - GeolengthelementsInput.z_first_crys / 2);
    m.rotate(90, 1.0, 0.0, 0.0);
    m.rotate(table_angle + c1_angle, 0.0, 1.0, 0.0);
    m.scale(GeolengthelementsInput.y_first_crys / 2, -table_posz / 2, GeolengthelementsInput.y_first_crys / 2);

    programShader->setUniformValue("matrix", m);
    pillarCylinderTexture->bind();
    drawObject(baseCylinderVertices, baseCylinderModelVertexBuffer, baseCylinderModelUVBuffer);

    m.scale(2 / GeolengthelementsInput.y_first_crys, 1 / (-table_posz / 2), 2 / GeolengthelementsInput.y_first_crys);
    m.rotate(-table_angle - c1_angle, 0.0, 1.0, 0.0);
    m.rotate(-90, 1.0, 0.0, 0.0);
    m.translate(0.0f, 0.0f, -table_posz / 2 + GeolengthelementsInput.z_first_crys / 2);


    //Crystal 2 Parallel
    m.rotate(table_angle, 0.0, 0.0, 1.0);
    m.translate(c2_posx, c2_posy, c2_posz);
    m.rotate(c2_angle_para, 0.0, 0.0, 1.0);
    m.rotate(GeoParametersInput.tilt_C2, 0.0f, 1.0f, 0.0f);
    m.scale(x_first_crys / 2, GeolengthelementsInput.y_first_crys / 2, GeolengthelementsInput.z_first_crys / 2);

    programShader->setUniformValue("matrix", m);
    crystalCubeTexture->bind();
    drawObject(baseCubeVertices, baseCubeModelVertexBuffer, baseCubeModelUVBuffer);

    m.scale(2 / x_first_crys, 2 / GeolengthelementsInput.y_first_crys, 2 / GeolengthelementsInput.z_first_crys);
    m.rotate(-GeoParametersInput.tilt_C2, 0.0f, 1.0f, 0.0f);
    m.rotate(-c2_angle_para, 0.0, 0.0, 1.0);
    m.translate(-c2_posx, -c2_posy, -c2_posz);
    m.rotate(-table_angle, 0.0, 0.0, 1.0);

    //Crystal 2 Parallel Pillar
    m.rotate(table_angle, 0.0, 0.0, 1.0);
    m.translate(c2_posx, c2_posy, c2_posz + table_posz / 2 - GeolengthelementsInput.z_first_crys / 2);
    m.rotate(c2_angle_para, 0.0, 0.0, 1.0);
    m.rotate(90, 1.0, 0.0, 0.0);
    m.rotate(table_angle + c1_angle, 0.0, 1.0, 0.0);
    m.scale(GeolengthelementsInput.y_first_crys / 2, -table_posz / 2, GeolengthelementsInput.y_first_crys / 2);

    programShader->setUniformValue("matrix", m);
    pillarCylinderTexture->bind();
    drawObject(baseCylinderVertices, baseCylinderModelVertexBuffer, baseCylinderModelUVBuffer);

    m.scale(2 / GeolengthelementsInput.y_first_crys, 1 / (-table_posz / 2), 2 / GeolengthelementsInput.y_first_crys);
    m.rotate(-table_angle - c1_angle, 0.0, 1.0, 0.0);
    m.rotate(-90, 1.0, 0.0, 0.0);
    m.rotate(-c2_angle_para, 0.0, 0.0, 1.0);
    m.translate(-c2_posx, -c2_posy, -c2_posz - table_posz / 2 + GeolengthelementsInput.z_first_crys / 2);
    m.rotate(-table_angle, 0.0, 0.0, 1.0);


    //Detector Parallel
    m.rotate(table_angle, 0.0, 0.0, 1.0);
    m.translate(c2_posx, c2_posy, c2_posz);
    m.rotate(detec_angle_para, 0.0, 0.0, 1.0);
    m.translate(detec_posx, detec_posy, detec_posz);
    m.scale(GeolengthelementsInput.zdetc / 2, GeolengthelementsInput.ydetc / 2, GeolengthelementsInput.zdetc / 2);

    programShader->setUniformValue("matrix", m);
    detecCylinderTexture->bind();
    drawObject(baseCylinderVertices, baseCylinderModelVertexBuffer, baseCylinderModelUVBuffer);

    m.scale(2 / GeolengthelementsInput.zdetc, 2 / GeolengthelementsInput.ydetc, 2 / GeolengthelementsInput.zdetc);
    m.translate(-detec_posx, -detec_posy, -detec_posz);
    m.rotate(-detec_angle_para, 0.0, 0.0, 1.0);
    m.translate(-c2_posx, -c2_posy, -c2_posz);
    m.rotate(-table_angle, 0.0, 0.0, 1.0);

    //Table
    m.rotate(table_angle, 0.0, 0.0, 1.0);
    m.translate(table_posx, table_posy, table_posz);
    m.scale(2 * (GeoParapathlengthsInput.dist_Cr1_Cr2 + GeoParapathlengthsInput.dist_Cr2_Det) / 3, 3 * GeolengthelementsInput.ydetc, GeolengthelementsInput.z_first_crys);

    programShader->setUniformValue("matrix", m);
    tableCubeTexture->bind();
    drawObject(baseCubeVertices, baseCubeModelVertexBuffer, baseCubeModelUVBuffer);

    m.scale(3 / (2 * (GeoParapathlengthsInput.dist_Cr1_Cr2 + GeoParapathlengthsInput.dist_Cr2_Det)), 1 / (3 * GeolengthelementsInput.ydetc), 1 / GeolengthelementsInput.z_first_crys);
    m.translate(-table_posx, -table_posy, -table_posz);
    m.rotate(-table_angle, 0.0, 0.0, 1.0);
    
}

void QDCS3Dvis::drawAntiParallel(QMatrix4x4 &m)
{
    source_posz -= 30.0f;
    ap_posz -= 30.0f;
    float c1_posz = -30.0f;
    c2_posz -= 30.0f;
    table_posz -= 30.0f;


    //Source Representation (i.e. entry flange to the chamber)
    m.translate(source_posx, source_posy, source_posz);
    m.scale((GeolengthelementsInput.S_sour + 1) / 2, S_sour_y / 2, (GeolengthelementsInput.S_sour + 1) / 2);

    programShader->setUniformValue("matrix", m);
    sourceCylinderTexture->bind();
    drawObject(baseCylinderVertices, baseCylinderModelVertexBuffer, baseCylinderModelUVBuffer);

    m.scale(2 / (GeolengthelementsInput.S_sour + 1), 2 / S_sour_y, 2 / (GeolengthelementsInput.S_sour + 1));
    m.translate(-source_posx, -source_posy, -source_posz);


    //Apperture Representation (Originally a Cu tube)
    m.translate(ap_posx, ap_posy, ap_posz);
    m.scale(GeolengthelementsInput.S_aper / 2, GeoParapathlengthsInput.LT_aper / 2, GeolengthelementsInput.S_aper / 2);

    programShader->setUniformValue("matrix", m);
    appertureCylinderTexture->bind();
    drawObject(baseCylinderVertices, baseCylinderModelVertexBuffer, baseCylinderModelUVBuffer);

    m.scale(2 / GeolengthelementsInput.S_aper, 2 / GeoParapathlengthsInput.LT_aper, 2 / GeolengthelementsInput.S_aper);
    m.translate(-ap_posx, -ap_posy, -ap_posz);


    table_angle = GeoParametersInput.teta_table + 90;
    c1_angle = GeoParametersInput.Exp_crys1 - GeoParametersInput.OffsetRotCry1;
    c2_angle_anti = table_angle + c1_angle + delrot * convdeg + 90;
    detec_angle_anti = GeoParametersInput.teta_detec_anti;


    //Crystal 1
    m.translate(0.0f, 0.0f, c1_posz);
    m.rotate(table_angle + c1_angle, 0.0, 0.0, 1.0);
    m.rotate(GeoParametersInput.tilt_C1, 0.0f, 1.0f, 0.0f);
    m.scale(x_first_crys / 2, GeolengthelementsInput.y_first_crys / 2, GeolengthelementsInput.z_first_crys / 2);

    programShader->setUniformValue("matrix", m);
    crystalCubeTexture->bind();
    drawObject(baseCubeVertices, baseCubeModelVertexBuffer, baseCubeModelUVBuffer);

    m.scale(2 / x_first_crys, 2 / GeolengthelementsInput.y_first_crys, 2 / GeolengthelementsInput.z_first_crys);
    m.rotate(-GeoParametersInput.tilt_C1, 0.0f, 1.0f, 0.0f);
    m.rotate(-table_angle - c1_angle, 0.0, 0.0, 1.0);
    m.translate(0.0f, 0.0f, -c1_posz);

    //Crystal 1 Pillar
    m.translate(0.0f, 0.0f, (c1_posz + table_posz) / 2 - GeolengthelementsInput.z_first_crys / 2);
    m.rotate(90, 1.0, 0.0, 0.0);
    m.rotate(table_angle + c1_angle, 0.0, 1.0, 0.0);
    m.scale(GeolengthelementsInput.y_first_crys / 2, -(table_posz - c1_posz) / 2, GeolengthelementsInput.y_first_crys / 2);

    programShader->setUniformValue("matrix", m);
    pillarCylinderTexture->bind();
    drawObject(baseCylinderVertices, baseCylinderModelVertexBuffer, baseCylinderModelUVBuffer);

    m.scale(2 / GeolengthelementsInput.y_first_crys, 1 / (-(table_posz - c1_posz) / 2), 2 / GeolengthelementsInput.y_first_crys);
    m.rotate(-table_angle - c1_angle, 0.0, 1.0, 0.0);
    m.rotate(-90, 1.0, 0.0, 0.0);
    m.translate(0.0f, 0.0f, -(c1_posz + table_posz) / 2 + GeolengthelementsInput.z_first_crys / 2);


    //Crystal 2 AntiParallel
    m.rotate(table_angle, 0.0, 0.0, 1.0);
    m.translate(c2_posx, c2_posy, c2_posz);
    m.rotate(c2_angle_anti, 0.0, 0.0, 1.0);
    m.rotate(GeoParametersInput.tilt_C2, 0.0f, 1.0f, 0.0f);
    m.scale(x_first_crys / 2, GeolengthelementsInput.y_first_crys / 2, GeolengthelementsInput.z_first_crys / 2);

    programShader->setUniformValue("matrix", m);
    crystalCubeTexture->bind();
    drawObject(baseCubeVertices, baseCubeModelVertexBuffer, baseCubeModelUVBuffer);

    m.scale(2 / x_first_crys, 2 / GeolengthelementsInput.y_first_crys, 2 / GeolengthelementsInput.z_first_crys);
    m.rotate(GeoParametersInput.tilt_C2, 0.0f, 1.0f, 0.0f);
    m.rotate(-c2_angle_anti, 0.0, 0.0, 1.0);
    m.translate(-c2_posx, -c2_posy, -c2_posz);
    m.rotate(-table_angle, 0.0, 0.0, 1.0);

    //Crystal 2 AntiParallel Pillar
    m.rotate(table_angle, 0.0, 0.0, 1.0);
    m.translate(c2_posx, c2_posy, (c2_posz + table_posz) / 2 - GeolengthelementsInput.z_first_crys / 2);
    m.rotate(c2_angle_anti, 0.0, 0.0, 1.0);
    m.rotate(90, 1.0, 0.0, 0.0);
    m.rotate(table_angle + c1_angle, 0.0, 1.0, 0.0);
    m.scale(GeolengthelementsInput.y_first_crys / 2, -(table_posz - c2_posz) / 2, GeolengthelementsInput.y_first_crys / 2);

    programShader->setUniformValue("matrix", m);
    pillarCylinderTexture->bind();
    drawObject(baseCylinderVertices, baseCylinderModelVertexBuffer, baseCylinderModelUVBuffer);

    m.scale(2 / GeolengthelementsInput.y_first_crys, 1 / (-(table_posz - c2_posz) / 2), 2 / GeolengthelementsInput.y_first_crys);
    m.rotate(-table_angle - c1_angle, 0.0, 1.0, 0.0);
    m.rotate(-90, 1.0, 0.0, 0.0);
    m.rotate(-c2_angle_anti, 0.0, 0.0, 1.0);
    m.translate(-c2_posx, -c2_posy, -(c2_posz + table_posz) / 2 + GeolengthelementsInput.z_first_crys / 2);
    m.rotate(-table_angle, 0.0, 0.0, 1.0);


    //Detector AntiParallel
    m.rotate(table_angle, 0.0, 0.0, 1.0);
    m.translate(c2_posx, c2_posy, c2_posz);
    m.rotate(detec_angle_anti, 0.0, 0.0, 1.0);
    m.translate(detec_posx, detec_posy, detec_posz);
    m.scale(GeolengthelementsInput.zdetc / 2, GeolengthelementsInput.ydetc / 2, GeolengthelementsInput.zdetc / 2);

    programShader->setUniformValue("matrix", m);
    detecCylinderTexture->bind();
    drawObject(baseCylinderVertices, baseCylinderModelVertexBuffer, baseCylinderModelUVBuffer);

    m.scale(2 / GeolengthelementsInput.zdetc, 2 / GeolengthelementsInput.ydetc, 2 / GeolengthelementsInput.zdetc);
    m.translate(-detec_posx, -detec_posy, -detec_posz);
    m.rotate(-detec_angle_anti, 0.0, 0.0, 1.0);
    m.translate(-c2_posx, -c2_posy, -c2_posz);
    m.rotate(-table_angle, 0.0, 0.0, 1.0);

    //Table
    m.rotate(table_angle, 0.0, 0.0, 1.0);
    m.translate(table_posx, table_posy, table_posz);
    m.scale(2 * (GeoParapathlengthsInput.dist_Cr1_Cr2 + GeoParapathlengthsInput.dist_Cr2_Det) / 3, 3 * GeolengthelementsInput.ydetc, GeolengthelementsInput.z_first_crys);

    programShader->setUniformValue("matrix", m);
    tableCubeTexture->bind();
    drawObject(baseCubeVertices, baseCubeModelVertexBuffer, baseCubeModelUVBuffer);

    m.scale(3 / (2 * (GeoParapathlengthsInput.dist_Cr1_Cr2 + GeoParapathlengthsInput.dist_Cr2_Det)), 1 / (3 * GeolengthelementsInput.ydetc), 1 / GeolengthelementsInput.z_first_crys);
    m.translate(-table_posx, -table_posy, -table_posz);
    m.rotate(-table_angle, 0.0, 0.0, 1.0);

    source_posz += 30.0f;
    ap_posz += 30.0f;
    c2_posz += 30.0f;
    table_posz += 30.0f;
}

void QDCS3Dvis::drawParallelText(QMatrix4x4 &m)
{

    //Source Representation (i.e. entry flange to the chamber)
    m.translate(source_posx, source_posy, source_posz);
    m.translate(0.0f, 0.0f, (GeolengthelementsInput.S_sour + 1) / 2 + text_voffset);
    m.rotate(90, 0.0, 0.0, 1.0);
    m.rotate(180, 1.0, 0.0, 0.0);
    m.scale(text_scale);

    programShader->setUniformValue("matrix", m);
    SourceTextTexture->bind();
    drawObject(SourceTextVertices, SourceTextModelVertexBuffer, SourceTextModelUVBuffer);

    m.scale(1 / text_scale);
    m.rotate(-180, 1.0, 0.0, 0.0);
    m.rotate(-90, 0.0, 0.0, 1.0);
    m.translate(0.0f, 0.0f, -(GeolengthelementsInput.S_sour + 1) / 2 - text_voffset);
    m.translate(-source_posx, -source_posy, -source_posz);


    //Apperture Representation (Originally a Cu tube)
    m.translate(ap_posx, ap_posy, ap_posz);
    m.translate(0.0f, 0.0f, GeolengthelementsInput.S_aper / 2 + text_voffset);
    m.rotate(90, 0.0, 0.0, 1.0);
    m.rotate(180, 1.0, 0.0, 0.0);
    m.scale(text_scale);

    programShader->setUniformValue("matrix", m);
    AppertureTextTexture->bind();
    drawObject(AppertureTextVertices, AppertureTextModelVertexBuffer, AppertureTextModelUVBuffer);

    m.scale(1 / text_scale);
    m.rotate(-180, 1.0, 0.0, 0.0);
    m.rotate(-90, 0.0, 0.0, 1.0);
    m.translate(0.0f, 0.0f, -GeolengthelementsInput.S_aper / 2 - text_voffset);
    m.translate(-ap_posx, -ap_posy, -ap_posz);


    table_angle = GeoParametersInput.teta_table + 90;
    c1_angle = GeoParametersInput.Exp_crys1 - GeoParametersInput.OffsetRotCry1;
    c2_angle_para = -90 - c1_angle + delrot * convdeg;
    detec_angle_para = GeoParametersInput.teta_detec_para;


    //Crystal 1
    m.translate(0.0f, 0.0f, GeolengthelementsInput.z_first_crys / 2 + text_voffset);
    m.rotate(90, 0.0, 0.0, 1.0);
    m.rotate(180, 1.0, 0.0, 0.0);
    m.scale(text_scale);

    programShader->setUniformValue("matrix", m);
    C1TextTexture->bind();
    drawObject(C1TextVertices, C1TextModelVertexBuffer, C1TextModelUVBuffer);

    m.scale(1 / text_scale);
    m.rotate(-180, 1.0, 0.0, 0.0);
    m.rotate(-90, 0.0, 0.0, 1.0);
    m.translate(0.0f, 0.0f, -GeolengthelementsInput.z_first_crys / 2 - text_voffset);


    //Crystal 2 Parallel
    m.rotate(table_angle, 0.0, 0.0, 1.0);
    m.translate(c2_posx, c2_posy, c2_posz);
    m.translate(0.0f, 0.0f, GeolengthelementsInput.z_first_crys / 2 + text_voffset);
    m.rotate(180, 1.0, 0.0, 0.0);
    m.scale(text_scale);

    programShader->setUniformValue("matrix", m);
    C2TextTexture->bind();
    drawObject(C2TextVertices, C2TextModelVertexBuffer, C2TextModelUVBuffer);

    m.scale(1 / text_scale);
    m.rotate(-180, 1.0, 0.0, 0.0);
    m.translate(0.0f, 0.0f, -GeolengthelementsInput.z_first_crys / 2 - text_voffset);
    m.translate(-c2_posx, -c2_posy, -c2_posz);
    m.rotate(-table_angle, 0.0, 0.0, 1.0);


    //Detector Parallel
    m.rotate(table_angle, 0.0, 0.0, 1.0);
    m.translate(c2_posx, c2_posy, c2_posz);
    m.rotate(detec_angle_para, 0.0, 0.0, 1.0);
    m.translate(detec_posx, detec_posy, detec_posz);
    m.translate(0.0f, 0.0f, GeolengthelementsInput.zdetc / 2 + text_voffset);
    m.rotate(180, 1.0, 0.0, 0.0);
    m.scale(text_scale);

    programShader->setUniformValue("matrix", m);
    DetectorTextTexture->bind();
    drawObject(DetectorTextVertices, DetectorTextModelVertexBuffer, DetectorTextModelUVBuffer);

    m.scale(1 / text_scale);
    m.rotate(-180, 1.0, 0.0, 0.0);
    m.translate(0.0f, 0.0f, -GeolengthelementsInput.zdetc / 2 - text_voffset);
    m.translate(-detec_posx, -detec_posy, -detec_posz);
    m.rotate(-detec_angle_para, 0.0, 0.0, 1.0);
    m.translate(-c2_posx, -c2_posy, -c2_posz);
    m.rotate(-table_angle, 0.0, 0.0, 1.0);

    //Table
    m.rotate(table_angle, 0.0, 0.0, 1.0);
    m.translate(table_posx, table_posy, table_posz);
    m.translate(0.0f, 0.0f, GeolengthelementsInput.z_first_crys + text_voffset);
    m.rotate(180, 1.0, 0.0, 0.0);
    m.scale(text_scale);

    programShader->setUniformValue("matrix", m);
    TableTextTexture->bind();
    drawObject(TableTextVertices, TableTextModelVertexBuffer, TableTextModelUVBuffer);

    m.scale(1 / text_scale);
    m.rotate(-180, 1.0, 0.0, 0.0);
    m.translate(0.0f, 0.0f, -GeolengthelementsInput.z_first_crys - text_voffset);
    m.translate(-table_posx, -table_posy, -table_posz);
    m.rotate(-table_angle, 0.0, 0.0, 1.0);

    //Parallel Configuration Text
    m.rotate(table_angle, 0.0, 0.0, 1.0);
    m.translate(table_posx, table_posy, table_posz);
    m.translate(-20 * text_voffset, 0.0f, -GeolengthelementsInput.z_first_crys - 3 * text_voffset);
    m.rotate(180, 1.0, 0.0, 0.0);
    m.scale(text_scale);

    programShader->setUniformValue("matrix", m);
    ParaTextTexture->bind();
    drawObject(ParaTextVertices, ParaTextModelVertexBuffer, ParaTextModelUVBuffer);

    m.scale(1 / text_scale);
    m.rotate(-180, 1.0, 0.0, 0.0);
    m.translate(20 * text_voffset, 0.0f, GeolengthelementsInput.z_first_crys + 3 * text_voffset);
    m.translate(-table_posx, -table_posy, -table_posz);
    m.rotate(-table_angle, 0.0, 0.0, 1.0);
}

void QDCS3Dvis::drawAntiParallelText(QMatrix4x4 &m)
{

    source_posz -= 30.0f;
    ap_posz -= 30.0f;
    float c1_posz = -30.0f;
    c2_posz -= 30.0f;
    table_posz -= 30.0f;

    //Source Representation (i.e. entry flange to the chamber)
    m.translate(source_posx, source_posy, source_posz);
    m.translate(0.0f, 0.0f, (GeolengthelementsInput.S_sour + 1) / 2 + text_voffset);
    m.rotate(90, 0.0, 0.0, 1.0);
    m.rotate(180, 1.0, 0.0, 0.0);
    m.scale(text_scale);

    programShader->setUniformValue("matrix", m);
    SourceTextTexture->bind();
    drawObject(SourceTextVertices, SourceTextModelVertexBuffer, SourceTextModelUVBuffer);

    m.scale(1 / text_scale);
    m.rotate(-180, 1.0, 0.0, 0.0);
    m.rotate(-90, 0.0, 0.0, 1.0);
    m.translate(0.0f, 0.0f, -(GeolengthelementsInput.S_sour + 1) / 2 - text_voffset);
    m.translate(-source_posx, -source_posy, -source_posz);


    //Apperture Representation (Originally a Cu tube)
    m.translate(ap_posx, ap_posy, ap_posz);
    m.translate(0.0f, 0.0f, GeolengthelementsInput.S_aper / 2 + text_voffset);
    m.rotate(90, 0.0, 0.0, 1.0);
    m.rotate(180, 1.0, 0.0, 0.0);
    m.scale(text_scale);

    programShader->setUniformValue("matrix", m);
    AppertureTextTexture->bind();
    drawObject(AppertureTextVertices, AppertureTextModelVertexBuffer, AppertureTextModelUVBuffer);

    m.scale(1 / text_scale);
    m.rotate(-180, 1.0, 0.0, 0.0);
    m.rotate(-90, 0.0, 0.0, 1.0);
    m.translate(0.0f, 0.0f, -GeolengthelementsInput.S_aper / 2 - text_voffset);
    m.translate(-ap_posx, -ap_posy, -ap_posz);


    table_angle = GeoParametersInput.teta_table + 90;
    c1_angle = GeoParametersInput.Exp_crys1 - GeoParametersInput.OffsetRotCry1;
    c2_angle_anti =  2 * table_angle + c1_angle + delrot * convdeg - 90;
    detec_angle_anti = GeoParametersInput.teta_detec_anti;


    //Crystal 1
    m.translate(0.0f, 0.0f, c1_posz);
    m.translate(0.0f, 0.0f, GeolengthelementsInput.z_first_crys / 2 + text_voffset);
    m.rotate(90, 0.0, 0.0, 1.0);
    m.rotate(180, 1.0, 0.0, 0.0);
    m.scale(text_scale);

    programShader->setUniformValue("matrix", m);
    C1TextTexture->bind();
    drawObject(C1TextVertices, C1TextModelVertexBuffer, C1TextModelUVBuffer);

    m.scale(1 / text_scale);
    m.rotate(-180, 1.0, 0.0, 0.0);
    m.rotate(-90, 0.0, 0.0, 1.0);
    m.translate(0.0f, 0.0f, -GeolengthelementsInput.z_first_crys / 2 - text_voffset);
    m.translate(0.0f, 0.0f, -c1_posz);


    //Crystal 2 AntiParallel
    m.rotate(table_angle, 0.0, 0.0, 1.0);
    m.translate(c2_posx, c2_posy, c2_posz);
    m.translate(0.0f, 0.0f, GeolengthelementsInput.z_first_crys / 2 + text_voffset);
    m.rotate(180, 1.0, 0.0, 0.0);
    m.scale(text_scale);

    programShader->setUniformValue("matrix", m);
    C2TextTexture->bind();
    drawObject(C2TextVertices, C2TextModelVertexBuffer, C2TextModelUVBuffer);

    m.scale(1 / text_scale);
    m.rotate(-180, 1.0, 0.0, 0.0);
    m.translate(0.0f, 0.0f, -GeolengthelementsInput.z_first_crys / 2 - text_voffset);
    m.translate(-c2_posx, -c2_posy, -c2_posz);
    m.rotate(-table_angle, 0.0, 0.0, 1.0);


    //Detector AntiParallel
    m.rotate(table_angle, 0.0, 0.0, 1.0);
    m.translate(c2_posx, c2_posy, c2_posz);
    m.rotate(detec_angle_anti, 0.0, 0.0, 1.0);
    m.translate(detec_posx, detec_posy, detec_posz);
    m.translate(0.0f, 0.0f, GeolengthelementsInput.zdetc / 2 + text_voffset);
    m.rotate(180, 1.0, 0.0, 0.0);
    m.scale(text_scale);

    programShader->setUniformValue("matrix", m);
    DetectorTextTexture->bind();
    drawObject(DetectorTextVertices, DetectorTextModelVertexBuffer, DetectorTextModelUVBuffer);

    m.scale(1 / text_scale);
    m.rotate(-180, 1.0, 0.0, 0.0);
    m.translate(0.0f, 0.0f, -GeolengthelementsInput.zdetc / 2 - text_voffset);
    m.translate(-detec_posx, -detec_posy, -detec_posz);
    m.rotate(-detec_angle_anti, 0.0, 0.0, 1.0);
    m.translate(-c2_posx, -c2_posy, -c2_posz);
    m.rotate(-table_angle, 0.0, 0.0, 1.0);

    //Table
    m.rotate(table_angle, 0.0, 0.0, 1.0);
    m.translate(table_posx, table_posy, table_posz);
    m.translate(0.0f, 0.0f, GeolengthelementsInput.z_first_crys + text_voffset);
    m.rotate(180, 1.0, 0.0, 0.0);
    m.scale(text_scale);

    programShader->setUniformValue("matrix", m);
    TableTextTexture->bind();
    drawObject(TableTextVertices, TableTextModelVertexBuffer, TableTextModelUVBuffer);

    m.scale(1 / text_scale);
    m.rotate(-180, 1.0, 0.0, 0.0);
    m.translate(0.0f, 0.0f, -GeolengthelementsInput.z_first_crys - text_voffset);
    m.translate(-table_posx, -table_posy, -table_posz);
    m.rotate(-table_angle, 0.0, 0.0, 1.0);

    //AntiParallel Configuration Text
    m.rotate(table_angle, 0.0, 0.0, 1.0);
    m.translate(table_posx, table_posy, table_posz);
    m.translate(-20 * text_voffset, 0.0f, -GeolengthelementsInput.z_first_crys - 3 * text_voffset);
    m.rotate(180, 1.0, 0.0, 0.0);
    m.scale(text_scale);

    programShader->setUniformValue("matrix", m);
    AntiTextTexture->bind();
    drawObject(AntiTextVertices, AntiTextModelVertexBuffer, AntiTextModelUVBuffer);

    m.scale(1 / text_scale);
    m.rotate(-180, 1.0, 0.0, 0.0);
    m.translate(20 * text_voffset, 0.0f, GeolengthelementsInput.z_first_crys + 3 * text_voffset);
    m.translate(-table_posx, -table_posy, -table_posz);
    m.rotate(-table_angle, 0.0, 0.0, 1.0);

    source_posz += 30.0f;
    ap_posz += 30.0f;
    c2_posz += 30.0f;
    table_posz += 30.0f;
}

void QDCS3Dvis::drawObject(std::vector<QVector3D> vertices, GLuint vbo, GLuint uvb)
{
    //Cube
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(
        0,                  // attribute
        3,                  // size
        GL_FLOAT,           // type
        GL_FALSE,           // normalized?
        0,                  // stride
        (void*)0            // array buffer offset
    );


    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, uvb);
    glVertexAttribPointer(
        1,                                // attribute
        2,                                // size
        GL_FLOAT,                         // type
        GL_FALSE,                         // normalized?
        0,                                // stride
        (void*)0                          // array buffer offset
    );


    glDrawArrays(GL_TRIANGLES, 0, (GLsizei)vertices.size() );

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
}


void QDCS3Dvis::drawParallelEvents(QMatrix4x4& m) {
    table_angle = GeoParametersInput.teta_table + 90;
    c1_angle = GeoParametersInput.Exp_crys1 - GeoParametersInput.OffsetRotCry1;
    c2_angle_para = -table_angle - c1_angle + delrot * convdeg + 90;
    detec_angle_para = GeoParametersInput.teta_detec_para;

    
    //Parallel events representation as yellow lines (similar to Geant4)
    for (std::vector<double> event : eventsToTrace_para) {
        
        if (event.size() > 5) {
            float sc1_HWDx = (event.at(4) + event.at(1)) / 4;
            float sc1_HWDz = (event.at(5) + event.at(2)) / 4;

            float sc1_anglex = atanf(2 * sc1_HWDx / abs(source_posy));
            float sc1_anglez = atanf(2 * sc1_HWDz / abs(source_posy));

            //Transform to the source reference frame
            m.translate(source_posx / 2 + sc1_HWDx, source_posy / 2 - event.at(0), source_posz + sc1_HWDz);
            m.rotate(sc1_anglex, 1.0, 0.0, 0.0);
            m.rotate(sc1_anglez, 0.0, 0.0, 1.0);
            m.scale(eventLineSize, source_posy / 2, eventLineSize);

            programShader->setUniformValue("matrix", m);
            sourceCylinderTexture->bind();
            drawObject(baseCylinderVertices, baseCylinderModelVertexBuffer, baseCylinderModelUVBuffer);

            m.scale(1 / eventLineSize, 2 / source_posy, 1 / eventLineSize);
            m.rotate(-sc1_anglex, 1.0, 0.0, 0.0);
            m.rotate(-sc1_anglez, 0.0, 0.0, 1.0);
            m.translate(-source_posx / 2 - sc1_HWDx, -source_posy / 2 + event.at(0), -source_posz - sc1_HWDz);


            if (event.size() > 6) {
                float c1c2_HWDy = (event.at(7) + event.at(4)) / 4;
                float c1c2_HWDz = (event.at(8) + event.at(5)) / 4;

                float c1c2_angley = atanf(2 * c1c2_HWDy / abs(c2_posx));
                float c1c2_anglez = atanf(2 * c1c2_HWDz / abs(c2_posx));

                //Transform to the second crystal reference frame in parallel
                m.rotate(table_angle, 0.0, 0.0, 1.0);
                m.translate(c2_posx / 2, c2_posy + c1c2_HWDy, c2_posz + c1c2_HWDz);
                m.rotate(90 + c1c2_anglez, 0.0, 0.0, 1.0);
                m.rotate(c1c2_angley, 0.0, 1.0, 0.0);
                m.scale(eventLineSize, c2_posx / 2, eventLineSize);

                programShader->setUniformValue("matrix", m);
                sourceCylinderTexture->bind();
                drawObject(baseCylinderVertices, baseCylinderModelVertexBuffer, baseCylinderModelUVBuffer);

                m.scale(1 / eventLineSize, 2 / c2_posx, 1 / eventLineSize);
                m.rotate(-90 - c1c2_anglez, 0.0, 0.0, 1.0);
                m.rotate(-c1c2_angley, 0.0, 1.0, 0.0);
                m.translate(-c2_posx / 2, -c2_posy - c1c2_HWDy, -c2_posz - c1c2_HWDz);
                m.rotate(-table_angle, 0.0, 0.0, 1.0);
            }

            if (event.size() > 9) {
                float c2det_HWDy = (event.at(10) + event.at(7)) / 4;
                float c2det_HWDz = (event.at(11) + event.at(8)) / 4;

                float c2det_angley = atanf(2 * c2det_HWDy / abs(detec_posx));
                float c2det_anglez = atanf(2 * c2det_HWDz / abs(detec_posx));

                //Transform to the detector reference frame in antiparallel
                m.rotate(table_angle, 0.0, 0.0, 1.0);
                m.translate(c2_posx, c2_posy, c2_posz);
                m.rotate(detec_angle_anti, 0.0, 0.0, 1.0);
                m.translate(detec_posx / 2, detec_posy + c2det_HWDy, detec_posz + c2det_HWDz);
                m.rotate(90 + c2det_anglez, 0.0, 0.0, 1.0);
                m.rotate(c2det_angley, 0.0, 1.0, 0.0);
                m.scale(eventLineSize, detec_posx / 2, eventLineSize);

                programShader->setUniformValue("matrix", m);
                sourceCylinderTexture->bind();
                drawObject(baseCylinderVertices, baseCylinderModelVertexBuffer, baseCylinderModelUVBuffer);

                m.scale(1 / eventLineSize, 2 / detec_posx, 1 / eventLineSize);
                m.rotate(-90 - c2det_anglez, 0.0, 0.0, 1.0);
                m.rotate(-c2det_angley, 0.0, 1.0, 0.0);
                m.translate(-detec_posx / 2, -detec_posy - c2det_HWDy, -detec_posz - c2det_HWDz);
                m.rotate(-detec_angle_anti, 0.0, 0.0, 1.0);
                m.translate(-c2_posx, -c2_posy, -c2_posz);
                m.rotate(-table_angle, 0.0, 0.0, 1.0);
            }
        }
    }
}

void QDCS3Dvis::drawAntiparallelEvents(QMatrix4x4& m) {
    source_posz -= 30.0f;
    ap_posz -= 30.0f;
    c2_posz -= 30.0f;
    table_posz -= 30.0f;

    
    table_angle = GeoParametersInput.teta_table + 90;
    c1_angle = GeoParametersInput.Exp_crys1 - GeoParametersInput.OffsetRotCry1;
    c2_angle_anti = table_angle + c1_angle + delrot * convdeg + 90;
    detec_angle_anti = GeoParametersInput.teta_detec_anti;

    //Antiparallel events representation as yellow lines (similar to Geant4)
    for (std::vector<double> event : eventsToTrace_anti) {

        if (event.size() > 5) {
            //if (event.at(1) > (GeolengthelementsInput.S_sour + 1) / 2) {
            //    std::cout << "erettr" << std::endl;
            //}
            float sc1_HWDx = (event.at(4) + event.at(1)) / 4;
            float sc1_HWDz = (event.at(5) + event.at(2)) / 4;

            float sc1_anglex = atanf(2 * sc1_HWDx / abs(source_posy));
            float sc1_anglez = atanf(2 * sc1_HWDz / abs(source_posy));

            //Transform to the source reference frame
            m.translate(source_posx / 2 + sc1_HWDx, source_posy / 2 + event.at(0), source_posz + sc1_HWDz);
            m.rotate(sc1_anglex, 1.0, 0.0, 0.0);
            m.rotate(sc1_anglez, 0.0, 0.0, 1.0);
            m.scale(eventLineSize, source_posy / 2, eventLineSize);

            programShader->setUniformValue("matrix", m);
            sourceCylinderTexture->bind();
            drawObject(baseCylinderVertices, baseCylinderModelVertexBuffer, baseCylinderModelUVBuffer);

            m.scale(1 / eventLineSize, 2 / source_posy, 1 / eventLineSize);
            m.rotate(-sc1_anglex, 1.0, 0.0, 0.0);
            m.rotate(-sc1_anglez, 0.0, 0.0, 1.0);
            m.translate(-source_posx / 2 - sc1_HWDx, -source_posy / 2 - event.at(0), -source_posz - sc1_HWDz);


            if (event.size() > 6) {
                float c1c2_HWDy = (event.at(7) + event.at(4)) / 4;
                float c1c2_HWDz = (event.at(8) + event.at(5)) / 4;

                float c1c2_angley = atanf(2 * c1c2_HWDy / abs(c2_posx));
                float c1c2_anglez = atanf(2 * c1c2_HWDz / abs(c2_posx));

                //Transform to the second crystal reference frame in parallel
                m.rotate(table_angle, 0.0, 0.0, 1.0);
                m.translate(c2_posx / 2, c2_posy + c1c2_HWDy, c2_posz + c1c2_HWDz);
                m.rotate(90 + c1c2_anglez, 0.0, 0.0, 1.0);
                m.rotate(c1c2_angley, 0.0, 1.0, 0.0);
                m.scale(eventLineSize, c2_posx / 2, eventLineSize);

                programShader->setUniformValue("matrix", m);
                sourceCylinderTexture->bind();
                drawObject(baseCylinderVertices, baseCylinderModelVertexBuffer, baseCylinderModelUVBuffer);

                m.scale(1 / eventLineSize, 2 / c2_posx, 1 / eventLineSize);
                m.rotate(-90 - c1c2_anglez, 0.0, 0.0, 1.0);
                m.rotate(-c1c2_angley, 0.0, 1.0, 0.0);
                m.translate(-c2_posx / 2, -c2_posy - c1c2_HWDy, -c2_posz - c1c2_HWDz);
                m.rotate(-table_angle, 0.0, 0.0, 1.0);
            }

            if (event.size() > 9) {
                float c2det_HWDy = (event.at(10) + event.at(7)) / 4;
                float c2det_HWDz = (event.at(11) + event.at(8)) / 4;

                float c2det_angley = atanf(2 * c2det_HWDy / abs(detec_posx));
                float c2det_anglez = atanf(2 * c2det_HWDz / abs(detec_posx));

                //Transform to the detector reference frame in antiparallel
                m.rotate(table_angle, 0.0, 0.0, 1.0);
                m.translate(c2_posx, c2_posy, c2_posz);
                m.rotate(detec_angle_anti, 0.0, 0.0, 1.0);
                m.translate(detec_posx / 2, detec_posy + c2det_HWDy, detec_posz + c2det_HWDz);
                m.rotate(90 + c2det_anglez, 0.0, 0.0, 1.0);
                m.rotate(c2det_angley, 0.0, 1.0, 0.0);
                m.scale(eventLineSize, detec_posx / 2, eventLineSize);

                programShader->setUniformValue("matrix", m);
                sourceCylinderTexture->bind();
                drawObject(baseCylinderVertices, baseCylinderModelVertexBuffer, baseCylinderModelUVBuffer);

                m.scale(1 / eventLineSize, 2 / detec_posx, 1 / eventLineSize);
                m.rotate(-90 - c2det_anglez, 0.0, 0.0, 1.0);
                m.rotate(-c2det_angley, 0.0, 1.0, 0.0);
                m.translate(-detec_posx / 2, -detec_posy - c2det_HWDy, -detec_posz - c2det_HWDz);
                m.rotate(-detec_angle_anti, 0.0, 0.0, 1.0);
                m.translate(-c2_posx, -c2_posy, -c2_posz);
                m.rotate(-table_angle, 0.0, 0.0, 1.0);
            }
        }
    }

    source_posz += 30.0f;
    ap_posz += 30.0f;
    c2_posz += 30.0f;
    table_posz += 30.0f;
}