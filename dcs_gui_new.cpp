#include "dcs_gui_new.h"
#include "ui_dcs_gui_new.h"

#include "simuGlobals.hh"
#include "simulationmain.h"
#include "wrapper/sserializer.inl"
#include <filesystem>
#include <fstream>
#include <QPushButton>
#include <QCheckBox>
#include <QRadioButton>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QComboBox>
#include <QString>
#include <QFileDialog>
#include <QTimer>

extern Geometry Geometry;
extern UserSettings UserSettings;
extern GeoParapathlengths GeoParapathlengths;
extern Geolengthelements Geolengthelements;
extern GeoParameters GeoParameters;
extern CurveVerticalTilt CurveVerticalTilt;
extern Graph_options Graph_options;
extern plotparameters plotparameters;
extern numberrays numberrays;
extern physical_parameters physical_parameters;
extern polarization_parameters polarization_parameters;
extern temperature_parameters temperature_parameters;
extern fullenergyspectrum fullenergyspectrum;
extern Curved_Crystal Curved_Crystal;
extern AnalysiesCrystaltilts AnalysiesCrystaltilts;

#ifdef LIB_DEF
extern double refra_corr;
extern std::string Unit_energy;
extern char File_simu[200];
extern char Output_dir[200];
#endif

static std::vector<uint8_t> serialize_inputs()
{
    std::vector<uint8_t> buffer;

    serialize_data(buffer, Geometry);

    serialize_data(buffer, UserSettings);

    serialize_data(buffer, GeoParapathlengths.type_source);
    serialize_data(buffer, GeoParapathlengths.LT_aper);
    serialize_data(buffer, GeoParapathlengths.dist_T_Cr1);
    serialize_data(buffer, GeoParapathlengths.dist_Cr1_Cr2);
    serialize_data(buffer, GeoParapathlengths.dist_Cr2_Det);
    serialize_data(buffer, GeoParapathlengths.dist_aper_det);

    serialize_data(buffer, Geolengthelements);

    serialize_data(buffer, GeoParameters);

    serialize_data(buffer, CurveVerticalTilt);

    serialize_data(buffer, Graph_options);

    serialize_data(buffer, plotparameters);

    serialize_data(buffer, numberrays);

    serialize_data(buffer, physical_parameters.Unit_energy);
    serialize_data(buffer, physical_parameters.linelamda);
    serialize_data(buffer, physical_parameters.naturalwidth);
    serialize_data(buffer, physical_parameters.gauss_Doop);

    serialize_data(buffer, polarization_parameters);

    serialize_data(buffer, temperature_parameters);

    // serialize_data(buffer, fullenergyspectrum);
    serialize_data(buffer, fullenergyspectrum.make_more_lines);
    serialize_data(buffer, fullenergyspectrum.energy_spectrum_file);
    serialize_data(buffer, fullenergyspectrum.linelamda1);
    serialize_data(buffer, fullenergyspectrum.naturalwidth1);
    serialize_data(buffer, fullenergyspectrum.p1_ener);
    serialize_data(buffer, fullenergyspectrum.linelamda2);
    serialize_data(buffer, fullenergyspectrum.naturalwidth2);
    serialize_data(buffer, fullenergyspectrum.p2_ener);
    serialize_data(buffer, fullenergyspectrum.linelamda3);
    serialize_data(buffer, fullenergyspectrum.naturalwidth3);
    serialize_data(buffer, fullenergyspectrum.p3_ener);
    serialize_data(buffer, fullenergyspectrum.linelamda4);
    serialize_data(buffer, fullenergyspectrum.naturalwidth4);
    serialize_data(buffer, fullenergyspectrum.Do_background);

    serialize_data(buffer, Curved_Crystal);

    serialize_data(buffer, AnalysiesCrystaltilts.make_matrix_full);
    serialize_data(buffer, AnalysiesCrystaltilts.make_graph_widths);
    serialize_data(buffer, AnalysiesCrystaltilts.metafile);
    serialize_data(buffer, AnalysiesCrystaltilts.make_an_C1_ta);
    serialize_data(buffer, AnalysiesCrystaltilts.make_plot_C1_table);

    return buffer;
}

static void deserialize_inputs(std::vector<uint8_t>& buffer)
{
    consume_buffer(buffer, Geometry);

    consume_buffer(buffer, UserSettings);

    consume_buffer(buffer, GeoParapathlengths.type_source);
    consume_buffer(buffer, GeoParapathlengths.LT_aper);
    consume_buffer(buffer, GeoParapathlengths.dist_T_Cr1);
    consume_buffer(buffer, GeoParapathlengths.dist_Cr1_Cr2);
    consume_buffer(buffer, GeoParapathlengths.dist_Cr2_Det);
    consume_buffer(buffer, GeoParapathlengths.dist_aper_det);

    consume_buffer(buffer, Geolengthelements);

    consume_buffer(buffer, GeoParameters);

    consume_buffer(buffer, CurveVerticalTilt);

    consume_buffer(buffer, Graph_options);

    consume_buffer(buffer, plotparameters);

    consume_buffer(buffer, numberrays);

    consume_buffer(buffer, physical_parameters.Unit_energy);
    consume_buffer(buffer, physical_parameters.linelamda);
    consume_buffer(buffer, physical_parameters.naturalwidth);
    consume_buffer(buffer, physical_parameters.gauss_Doop);

    consume_buffer(buffer, polarization_parameters);

    consume_buffer(buffer, temperature_parameters);

    // consume_buffer(buffer, fullenergyspectrum);
    consume_buffer(buffer, fullenergyspectrum.make_more_lines);
    consume_buffer(buffer, fullenergyspectrum.energy_spectrum_file);
    consume_buffer(buffer, fullenergyspectrum.linelamda1);
    consume_buffer(buffer, fullenergyspectrum.naturalwidth1);
    consume_buffer(buffer, fullenergyspectrum.p1_ener);
    consume_buffer(buffer, fullenergyspectrum.linelamda2);
    consume_buffer(buffer, fullenergyspectrum.naturalwidth2);
    consume_buffer(buffer, fullenergyspectrum.p2_ener);
    consume_buffer(buffer, fullenergyspectrum.linelamda3);
    consume_buffer(buffer, fullenergyspectrum.naturalwidth3);
    consume_buffer(buffer, fullenergyspectrum.p3_ener);
    consume_buffer(buffer, fullenergyspectrum.linelamda4);
    consume_buffer(buffer, fullenergyspectrum.naturalwidth4);
    consume_buffer(buffer, fullenergyspectrum.Do_background);

    consume_buffer(buffer, Curved_Crystal);

    consume_buffer(buffer, AnalysiesCrystaltilts.make_matrix_full);
    consume_buffer(buffer, AnalysiesCrystaltilts.make_graph_widths);
    consume_buffer(buffer, AnalysiesCrystaltilts.metafile);
    consume_buffer(buffer, AnalysiesCrystaltilts.make_an_C1_ta);
    consume_buffer(buffer, AnalysiesCrystaltilts.make_plot_C1_table);
}

static void write_buffer_to_file(const std::vector<uint8_t>& buffer, const std::string& filename)
{
    std::ofstream stream(filename.c_str(), std::ios::binary);
    if(!stream.is_open())
    {
        std::cout << "Failed to write buffer to file." << std::endl;
        return;
    }

    stream.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
}

static std::vector<uint8_t> read_buffer_from_file(const std::string& filename)
{
    std::vector<uint8_t> buffer;

    std::ifstream stream(filename.c_str(), std::ios::binary);
    if(!stream.is_open())
    {
        std::cout << "Failed to read buffer from file." << std::endl;
        return buffer;
    }

    stream.unsetf(std::ios::skipws);

    stream.seekg(0, std::ios::end);
    std::streampos filesize = stream.tellg();
    stream.seekg(0, std::ios::beg);

    buffer.reserve(filesize);

    std::copy(std::istream_iterator<uint8_t>(stream), std::istream_iterator<uint8_t>(), std::back_inserter(buffer));

    assert(buffer.size() > 0);

    return buffer;
}

GUISettingsWindow::GUISettingsWindow(QWidget *parent) : 
    QMainWindow(parent),
    ui(new Ui::GUISettingsWindow),
    wsimu(new SimulationMain()),
    timer(new QTimer(this))
{
    ui->setupUi(this);

    // Setup all of the connections to the settings

    // ================================================================================================
    // User Settings
    // ================================================================================================
    connect(ui->make_parallel, &QCheckBox::stateChanged, this, [this](int s) { UserSettings.see_para = (bool)s; });
    connect(ui->make_antiparallel, &QCheckBox::stateChanged, this, [this](int s) { UserSettings.see_anti = (bool)s; });
    connect(ui->vertical_divergence, &QCheckBox::stateChanged, this, [this](int s) { UserSettings.Make_Vertical = (bool)s; });
    connect(ui->horizontal_divergence, &QCheckBox::stateChanged, this, [this](int s) { UserSettings.Make_Horizontal = (bool)s; });
    connect(ui->fitting, &QCheckBox::stateChanged, this, [this](int s) { UserSettings.fitting = (bool)s; });
    connect(ui->fitting_truev, &QCheckBox::stateChanged, this, [this](int s) { UserSettings.TrueVoigt = (bool)s; });
    connect(ui->center_c1, &QCheckBox::stateChanged, this, [this](int s) { UserSettings.center_1crys = (bool)s; });
    connect(ui->center_c2, &QCheckBox::stateChanged, this, [this](int s) { UserSettings.center_2crys = (bool)s; });
    connect(ui->print_scan, &QCheckBox::stateChanged, this, [this](int s) { UserSettings.print_scan = (bool)s; });
    connect(ui->center_mask, &QCheckBox::stateChanged, this, [this](int s) { UserSettings.center_Mask = (bool)s; });
    connect(ui->test_mask, &QCheckBox::stateChanged, this, [this](int s) { UserSettings.make_mask_test = (bool)s; });
    connect(ui->aa_none, &QRadioButton::clicked, this, [this]() { UserSettings.angle_aprox = 0; });
    connect(ui->aa_partial, &QRadioButton::clicked, this, [this]() { UserSettings.angle_aprox = 1; });
    connect(ui->aa_full, &QRadioButton::clicked, this, [this]() { UserSettings.angle_aprox = 2; });
    connect(ui->c1m_no, &QRadioButton::clicked, this, [this]() { UserSettings.mask_C1 = 0; });
    connect(ui->c1m_down, &QRadioButton::clicked, this, [this]() { UserSettings.mask_C1 = 1; });
    connect(ui->c1m_up, &QRadioButton::clicked, this, [this]() { UserSettings.mask_C1 = 2; });
    connect(ui->c2m_no, &QRadioButton::clicked, this, [this]() { UserSettings.mask_C2 = 0; });
    connect(ui->c2m_down, &QRadioButton::clicked, this, [this]() { UserSettings.mask_C2 = 1; });
    connect(ui->c2m_up, &QRadioButton::clicked, this, [this]() { UserSettings.mask_C2 = 2; });
    connect(ui->st_simple, &QRadioButton::clicked, this, [this]() { UserSettings.Simple_simu = true; });
    connect(ui->st_complex, &QRadioButton::clicked, this, [this]() { UserSettings.Simple_simu = false; });

    // ================================================================================================
    // Geometric Settings
    // ================================================================================================
    connect(ui->spinBox_5, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int v) { Geometry.imh = v; });
    connect(ui->spinBox_6, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int v) { Geometry.imk = v; });
    connect(ui->spinBox_7, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int v) { Geometry.iml = v; });
    // Geometry mode (reserved)
    // Crystal type (reserved)
    connect(ui->src_geometry, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int v) { 
        switch(v)
        {
            case 0: GeoParapathlengths.type_source = "UC"; break;
            case 1: GeoParapathlengths.type_source = "UR"; break;
            case 2: GeoParapathlengths.type_source =  "P"; break;
            case 3: GeoParapathlengths.type_source =  "G"; break;
            default: break;
        }
    });
    connect(ui->src_aprt_11, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { Geolengthelements.S_sour = v; });
    connect(ui->src_aprt_12, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { Geolengthelements.S_aper = v; });
    connect(ui->src_aprt_18, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { Geolengthelements.y_sour = v; });
    connect(ui->src_aprt_19, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { Geolengthelements.z_sour = v; });
    connect(ui->src_aprt_20, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { Geolengthelements.y_aper = v; });
    connect(ui->src_aprt_21, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { Geolengthelements.z_aper = v; });
    connect(ui->src_aprt_13, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { Geolengthelements.S_aper_var = v; });
    connect(ui->src_aprt_14, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { Geolengthelements.S_shi_hor_A = v; });
    connect(ui->src_aprt_15, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { Geolengthelements.S_shi_ver_A = v; });
    connect(ui->src_aprt_16, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { Geolengthelements.S_shi_hor_B = v; });
    connect(ui->src_aprt_17, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { Geolengthelements.S_shi_ver_B = v; });
    connect(ui->src_aprt   , QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { GeoParapathlengths.LT_aper = v; });
    connect(ui->src_aprt_2 , QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { GeoParapathlengths.dist_T_Cr1 = v; });
    connect(ui->src_aprt_3 , QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { GeoParapathlengths.dist_Cr1_Cr2 = v; });
    connect(ui->src_aprt_4 , QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { GeoParapathlengths.dist_Cr2_Det = v; });
    connect(ui->src_aprt_6 , QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { Geolengthelements.y_first_crys = v; });
    connect(ui->src_aprt_7 , QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { Geolengthelements.z_first_crys = v; });
    connect(ui->src_aprt_8 , QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { Geolengthelements.ydetc = v; });
    connect(ui->src_aprt_9 , QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { Geolengthelements.zdetc = v; });
    connect(ui->src_aprt_10, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { Geolengthelements.shift_det_ver = v; });

    // ================================================================================================
    // Crystal Settings
    // ================================================================================================
    connect(ui->src_aprt_22, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { GeoParameters.Exp_crys1 = v; });
    connect(ui->src_aprt_23, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { GeoParameters.teta_table = v; });
    connect(ui->src_aprt_24, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { GeoParameters.OffsetRotCry1 = v; });
    connect(ui->src_aprt_25, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { GeoParameters.teta_detec_para = v; });
    connect(ui->src_aprt_26, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { GeoParameters.teta_detec_anti = v; });
    connect(ui->src_aprt_27, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { GeoParameters.tilt_C1 = v; });
    connect(ui->src_aprt_29, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { GeoParameters.tilt_C2 = v; });
    connect(ui->src_aprt_30, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { GeoParameters.xsi = v; });
    connect(ui->src_aprt_31, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { GeoParameters.center_1cry_at = v; });
    connect(ui->src_aprt_32, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { GeoParameters.center_2cry_at = v; });
    connect(ui->src_aprt_33, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { CurveVerticalTilt.phas_tilt1 = v; });
    connect(ui->src_aprt_34, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { CurveVerticalTilt.phas_tilt2 = v; });
    connect(ui->src_aprt_35, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { CurveVerticalTilt.offsettilt1 = v; });
    connect(ui->src_aprt_36, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { CurveVerticalTilt.offsettilt2 = v; });
    connect(ui->src_aprt_37, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { CurveVerticalTilt.consttilt1 = v; });
    connect(ui->src_aprt_38, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { CurveVerticalTilt.consttilt2 = v; });
    connect(ui->make_img_plates_6, &QCheckBox::stateChanged, this, [this](int s) { Curved_Crystal.Curve_crystall = (bool)s; });
    connect(ui->src_aprt_53, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { Curved_Crystal.R_cur_crys_1 = v; });
    connect(ui->src_aprt_54, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { Curved_Crystal.R_cur_crys_2 = v; });

    // ================================================================================================
    // Plot Settings
    // ================================================================================================
    connect(ui->make_img_plates_5, &QCheckBox::stateChanged, this, [this](int s) { Graph_options.make_graph_profile = (bool)s; });
    connect(ui->make_img_plates  , &QCheckBox::stateChanged, this, [this](int s) { Graph_options.make_image_plates = (bool)s; });
    connect(ui->make_img_plates_3, &QCheckBox::stateChanged, this, [this](int s) { Graph_options.make_imageC1_After_refle = (bool)s; });
    connect(ui->make_img_plates_4, &QCheckBox::stateChanged, this, [this](int s) { Graph_options.make_imageC2_After_refle = (bool)s; });
    connect(ui->src_aprt_40, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { plotparameters.delta_angl = v; });
    connect(ui->src_aprt_41, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { plotparameters.shift_disp_window = v; });
    connect(ui->spinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int v) { plotparameters.nubins = v; });

    // ================================================================================================
    // Other Settings
    // ================================================================================================
    connect(ui->spinBox_2, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int v) { numberrays.nbeams = v; });
    connect(ui->spinBox_3, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int v) { numberrays.number_rotati = v; });
    connect(ui->src_geometry_3, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int v) { 
        switch(v)
        {
            case 0: physical_parameters.Unit_energy =  "eV"; break;
            case 1: physical_parameters.Unit_energy = "keV"; break;
            case 2: physical_parameters.Unit_energy =   "A"; break;
            default: break;
        }
    });
    connect(ui->src_aprt_42, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { physical_parameters.linelamda = v; });
    connect(ui->src_aprt_43, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { physical_parameters.naturalwidth = v; });
    connect(ui->src_aprt_45, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { physical_parameters.gauss_Doop = v; });
    connect(ui->src_geometry_4, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int v) { 
        switch(v)
        {
            case 0: 
                fullenergyspectrum.make_more_lines = 2;
                ui->src_aprt_55->setEnabled(false);
                ui->src_aprt_56->setEnabled(false);
                ui->src_aprt_57->setEnabled(false);
                ui->src_aprt_58->setEnabled(false);
                ui->src_aprt_59->setEnabled(false);
                ui->src_aprt_60->setEnabled(false);
                ui->src_aprt_61->setEnabled(false);
                ui->src_aprt_62->setEnabled(false);
                ui->src_aprt_64->setEnabled(false);
                ui->src_aprt_65->setEnabled(false);
                ui->src_aprt_66->setEnabled(false);
                ui->lineEdit->setEnabled(true);
                ui->pushButton->setEnabled(true);
                break;
            case 1: 
                fullenergyspectrum.make_more_lines = 1;
                ui->src_aprt_55->setEnabled(true);
                ui->src_aprt_56->setEnabled(true);
                ui->src_aprt_57->setEnabled(true);
                ui->src_aprt_58->setEnabled(true);
                ui->src_aprt_59->setEnabled(true);
                ui->src_aprt_60->setEnabled(true);
                ui->src_aprt_61->setEnabled(true);
                ui->src_aprt_62->setEnabled(true);
                ui->src_aprt_64->setEnabled(true);
                ui->src_aprt_65->setEnabled(true);
                ui->src_aprt_66->setEnabled(true);
                ui->lineEdit->setEnabled(false);
                ui->pushButton->setEnabled(false);
                break;
            case 2: 
                fullenergyspectrum.make_more_lines = 0;
                ui->src_aprt_55->setEnabled(true);
                ui->src_aprt_56->setEnabled(true);
                ui->src_aprt_57->setEnabled(false);
                ui->src_aprt_58->setEnabled(false);
                ui->src_aprt_59->setEnabled(false);
                ui->src_aprt_60->setEnabled(false);
                ui->src_aprt_61->setEnabled(false);
                ui->src_aprt_62->setEnabled(false);
                ui->src_aprt_64->setEnabled(false);
                ui->src_aprt_65->setEnabled(false);
                ui->src_aprt_66->setEnabled(false);
                ui->lineEdit->setEnabled(false);
                ui->pushButton->setEnabled(false);
                break;
            default: break;
        }
    });
    connect(ui->pushButton, &QPushButton::clicked, this, [this]() {
        QString filename = QFileDialog::getOpenFileName(this,
            tr("Open Spectrum Data"),
            QString(),
            tr("Text Documents (*.txt)")
        );

        if(filename.isEmpty())
        {
            return;
        }

        fullenergyspectrum.energy_spectrum_file = filename.toStdString();
        ui->lineEdit->setText(QString::fromStdString(fullenergyspectrum.energy_spectrum_file));
    });
    connect(ui->src_aprt_55, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { fullenergyspectrum.linelamda1 = v; });
    connect(ui->src_aprt_56, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { fullenergyspectrum.naturalwidth1 = v; });
    connect(ui->src_aprt_57, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { fullenergyspectrum.p1_ener = v; });
    connect(ui->src_aprt_58, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { fullenergyspectrum.linelamda2 = v; });
    connect(ui->src_aprt_59, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { fullenergyspectrum.naturalwidth2 = v; });
    connect(ui->src_aprt_60, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { fullenergyspectrum.p2_ener = v; });
    connect(ui->src_aprt_61, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { fullenergyspectrum.linelamda3 = v; });
    connect(ui->src_aprt_62, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { fullenergyspectrum.naturalwidth3 = v; });
    connect(ui->src_aprt_64, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { fullenergyspectrum.p3_ener = v; });
    connect(ui->src_aprt_65, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { fullenergyspectrum.linelamda4 = v; });
    connect(ui->src_aprt_66, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { fullenergyspectrum.naturalwidth4 = v; });
    connect(ui->make_bg, &QCheckBox::stateChanged, this, [this](int s) { fullenergyspectrum.Do_background = (bool)s; });
    connect(ui->make_pol, &QCheckBox::stateChanged, this, [this](int s) { polarization_parameters.mka_poli = (bool)s; });
    connect(ui->src_aprt_46, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { polarization_parameters.relationP_S = v; });

    // ================================================================================================
    // Temperature Settings
    // ================================================================================================
    connect(ui->src_aprt_47, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { temperature_parameters.T_crystal_1_para = v; });
    connect(ui->src_aprt_48, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { temperature_parameters.T_crystal_2_para = v; });
    connect(ui->src_aprt_50, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { temperature_parameters.T_crystal_1_anti = v; });
    connect(ui->src_aprt_49, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { temperature_parameters.T_crystal_2_anti = v; });
    connect(ui->make_temp_change, &QCheckBox::stateChanged, this, [this](int s) { temperature_parameters.mk_temp_bin = (bool)s; });
    connect(ui->src_aprt_51, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { temperature_parameters.AA_tempera = v; });
    connect(ui->src_aprt_52, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { temperature_parameters.TT_tempera = v; });

    updateElements();

    // Buttons
    connect(ui->saveSim_button, &QPushButton::clicked, this, [this]() {
        QString filename = QFileDialog::getSaveFileName(this,
            tr("Save Simulation Settings"),
            tr("simu_settings.dsp"),
            tr("DCSsimu params (*.dsp)")
        );

        if(filename.isEmpty())
        {
            return;
        }

        auto buffer = serialize_inputs();
        write_buffer_to_file(buffer, filename.toStdString());
    });

    connect(ui->loadSim_button, &QPushButton::clicked, this, [this]() {
        QString filename = QFileDialog::getOpenFileName(this,
            tr("Open Simulation Settings"),
            QString(),
            tr("DCSsimu params (*.dsp)")
        );

        if(filename.isEmpty())
        {
            return;
        }

        auto buffer = read_buffer_from_file(filename.toStdString());
        deserialize_inputs(buffer);

        updateElements();

#ifdef LIB_DEF
        // Load OK
        // TODO(César): Setting the options manually should also enable the start button 
        ui->startSim_button->setEnabled(true);
        ui->startSim_button->setStyleSheet(QString(
            "border-color: rgb(0, 0, 0);"
            "background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0,stop:0 rgba(0, 100, 0, 255), stop:1 rgba(0, 170, 0, 255));"
        ));
#endif
    });

#ifdef LIB_DEF
    ui->startSim_button->setEnabled(false);
    ui->startSim_button->setStyleSheet(QString(
        "border-color: rgb(0, 0, 0);"
        "background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0,stop:0 rgba(100, 100, 100, 255), stop:1 rgba(100, 100, 100, 255));"
    ));

    connect(ui->startSim_button, &QPushButton::clicked, wsimu, &SimulationMain::startSimulationThread);
    connect(ui->startSim_button, &QPushButton::clicked, this, [this]() { timer->start(100); });
    connect(timer, &QTimer::timeout, this, [this]() {
        float pct = wsimu->getPctDone();
        ui->startSim_button->setStyleSheet(QString(
            "border-color: rgb(0, 0, 0);"
            "background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0,"
            "stop:0 rgba(0, 100, 0, 255), stop:%1 rgba(0, 170, 0, 255),"
            "stop:%2 rgba(100, 100, 100, 255), stop:1 rgba(100, 100, 100, 255));"
        ).arg(pct-0.02f).arg(pct));
        int ipct = static_cast<int>(pct * 100.0f);
        ui->startSim_button->setText(QString::number(ipct) + "%");

        if(pct >= 0.999f)
        {
            timer->stop();
            ui->startSim_button->setStyleSheet(QString(
                "border-color: rgb(0, 0, 0);"
                "background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0,stop:0 rgba(0, 100, 0, 255), stop:1 rgba(0, 170, 0, 255));"
            ));
            ui->startSim_button->setText("Start Simulation");
        }
    });

    // Where do we output to when using the lib?
    strcat(Output_dir, "DCSsimu_output");
    if(!std::filesystem::is_directory(Output_dir) || !std::filesystem::exists(Output_dir))
    {
        std::filesystem::create_directories(Output_dir);
    }

    // Where does the workspace live for the lib?
#ifndef LIB_DATA_DIR
#error LIB_DATA_DIR must be defined within a library target build.
#endif
    strcpy(File_simu, LIB_DATA_DIR);

    // Make aditional configurations
    Unit_energy = physical_parameters.Unit_energy;
    if(Geometry.imh == 2 and Geometry.imk == 2 and Geometry.iml == 2)
    {
        refra_corr = refra_corrNIST;
    }
    else
    {
        refra_corr = refra_corrPARIS;
    }

#else
    connect(ui->startSim_button, &QPushButton::clicked, wsimu, &SimulationMain::show);
#endif
}

GUISettingsWindow::~GUISettingsWindow()
{
    delete ui;
}

void GUISettingsWindow::updateElements()
{
    // ================================================================================================
    // User Settings
    // ================================================================================================
    ui->make_parallel->setChecked(UserSettings.see_para);
    ui->make_antiparallel->setChecked(UserSettings.see_anti);
    ui->vertical_divergence->setChecked(UserSettings.Make_Vertical);
    ui->horizontal_divergence->setChecked(UserSettings.Make_Horizontal);
    ui->fitting->setChecked(UserSettings.fitting);
    ui->fitting_truev->setChecked(UserSettings.TrueVoigt);
    ui->center_c1->setChecked(UserSettings.center_1crys);
    ui->center_c2->setChecked(UserSettings.center_2crys);
    ui->print_scan->setChecked(UserSettings.print_scan);
    ui->center_mask->setChecked(UserSettings.center_Mask);
    ui->test_mask->setChecked(UserSettings.make_mask_test);
    ui->aa_none->setChecked(UserSettings.angle_aprox == 0);
    ui->aa_partial->setChecked(UserSettings.angle_aprox == 1);
    ui->aa_full->setChecked(UserSettings.angle_aprox == 2);
    ui->c1m_no->setChecked(UserSettings.mask_C1 == 0);
    ui->c1m_down->setChecked(UserSettings.mask_C1 == 1);
    ui->c1m_up->setChecked(UserSettings.mask_C1 == 2);
    ui->c2m_no->setChecked(UserSettings.mask_C2 == 0);
    ui->c2m_down->setChecked(UserSettings.mask_C2 == 1);
    ui->c2m_up->setChecked(UserSettings.mask_C2 == 2);
    ui->st_simple->setChecked(UserSettings.Simple_simu);
    ui->st_complex->setChecked(!UserSettings.Simple_simu);

    // ================================================================================================
    // Geometric Settings
    // ================================================================================================
    ui->spinBox_5->setValue(Geometry.imh);
    ui->spinBox_6->setValue(Geometry.imk);
    ui->spinBox_7->setValue(Geometry.iml);
    // Geometry mode (reserved)
    // Crystal type (reserved)
    if(GeoParapathlengths.type_source == "UC")      ui->src_geometry->setCurrentIndex(0);
    else if(GeoParapathlengths.type_source == "UR") ui->src_geometry->setCurrentIndex(1);
    else if(GeoParapathlengths.type_source ==  "P") ui->src_geometry->setCurrentIndex(2);
    else if(GeoParapathlengths.type_source ==  "G") ui->src_geometry->setCurrentIndex(3);
    ui->src_aprt_11->setValue(Geolengthelements.S_sour);
    ui->src_aprt_12->setValue(Geolengthelements.S_aper);
    ui->src_aprt_18->setValue(Geolengthelements.y_sour);
    ui->src_aprt_19->setValue(Geolengthelements.z_sour);
    ui->src_aprt_20->setValue(Geolengthelements.y_aper);
    ui->src_aprt_21->setValue(Geolengthelements.z_aper);
    ui->src_aprt_13->setValue(Geolengthelements.S_aper_var);
    ui->src_aprt_14->setValue(Geolengthelements.S_shi_hor_A);
    ui->src_aprt_15->setValue(Geolengthelements.S_shi_ver_A);
    ui->src_aprt_16->setValue(Geolengthelements.S_shi_hor_B);
    ui->src_aprt_17->setValue(Geolengthelements.S_shi_ver_B);
    ui->src_aprt->setValue(GeoParapathlengths.LT_aper);
    ui->src_aprt_2->setValue(GeoParapathlengths.dist_T_Cr1);
    ui->src_aprt_3->setValue(GeoParapathlengths.dist_Cr1_Cr2);
    ui->src_aprt_4->setValue(GeoParapathlengths.dist_Cr2_Det);
    ui->src_aprt_6->setValue(Geolengthelements.y_first_crys);
    ui->src_aprt_7->setValue(Geolengthelements.z_first_crys);
    ui->src_aprt_8->setValue(Geolengthelements.ydetc);
    ui->src_aprt_9->setValue(Geolengthelements.zdetc);
    ui->src_aprt_10->setValue(Geolengthelements.shift_det_ver);

    // ================================================================================================
    // Crystal Settings
    // ================================================================================================
    ui->src_aprt_22->setValue(GeoParameters.Exp_crys1);
    ui->src_aprt_23->setValue(GeoParameters.teta_table);
    ui->src_aprt_24->setValue(GeoParameters.OffsetRotCry1);
    ui->src_aprt_25->setValue(GeoParameters.teta_detec_para);
    ui->src_aprt_26->setValue(GeoParameters.teta_detec_anti);
    ui->src_aprt_27->setValue(GeoParameters.tilt_C1);
    ui->src_aprt_29->setValue(GeoParameters.tilt_C2);
    ui->src_aprt_30->setValue(GeoParameters.xsi);
    ui->src_aprt_31->setValue(GeoParameters.center_1cry_at);
    ui->src_aprt_32->setValue(GeoParameters.center_2cry_at);
    ui->src_aprt_33->setValue(CurveVerticalTilt.phas_tilt1);
    ui->src_aprt_34->setValue(CurveVerticalTilt.phas_tilt2);
    ui->src_aprt_35->setValue(CurveVerticalTilt.offsettilt1);
    ui->src_aprt_36->setValue(CurveVerticalTilt.offsettilt2);
    ui->src_aprt_37->setValue(CurveVerticalTilt.consttilt1);
    ui->src_aprt_38->setValue(CurveVerticalTilt.consttilt2);
    ui->make_img_plates_6->setChecked(Curved_Crystal.Curve_crystall);
    ui->src_aprt_53->setValue(Curved_Crystal.R_cur_crys_1);
    ui->src_aprt_54->setValue(Curved_Crystal.R_cur_crys_2);

    // ================================================================================================
    // Plot Settings
    // ================================================================================================
    ui->make_img_plates_5->setChecked(Graph_options.make_graph_profile);
    ui->make_img_plates->setChecked(Graph_options.make_image_plates);
    ui->make_img_plates_3->setChecked(Graph_options.make_imageC1_After_refle);
    ui->make_img_plates_4->setChecked(Graph_options.make_imageC2_After_refle);
    ui->src_aprt_40->setValue(plotparameters.delta_angl);
    ui->src_aprt_41->setValue(plotparameters.shift_disp_window);
    ui->spinBox->setValue(plotparameters.nubins);

    // ================================================================================================
    // Other Settings
    // ================================================================================================
    ui->spinBox_2->setValue(numberrays.nbeams);
    ui->spinBox_3->setValue(numberrays.number_rotati);
    if(physical_parameters.Unit_energy ==  "eV")      ui->src_geometry_3->setCurrentIndex(0);
    else if(physical_parameters.Unit_energy == "keV") ui->src_geometry_3->setCurrentIndex(1);
    else if(physical_parameters.Unit_energy ==   "A") ui->src_geometry_3->setCurrentIndex(2);
    ui->src_aprt_42->setValue(physical_parameters.linelamda);
    ui->src_aprt_43->setValue(physical_parameters.naturalwidth);
    ui->src_aprt_45->setValue(physical_parameters.gauss_Doop);
    switch(fullenergyspectrum.make_more_lines)
    {
        case 2: ui->src_geometry_4->setCurrentIndex(0); emit ui->src_geometry_4->currentIndexChanged(0); break;
        case 1: ui->src_geometry_4->setCurrentIndex(1); emit ui->src_geometry_4->currentIndexChanged(0); break;
        case 0: ui->src_geometry_4->setCurrentIndex(2); emit ui->src_geometry_4->currentIndexChanged(0); break;
        default: break;
    }
    ui->lineEdit->setText(QString::fromStdString(fullenergyspectrum.energy_spectrum_file));
    ui->src_aprt_55->setValue(fullenergyspectrum.linelamda1);
    ui->src_aprt_56->setValue(fullenergyspectrum.naturalwidth1);
    ui->src_aprt_57->setValue(fullenergyspectrum.p1_ener);
    ui->src_aprt_58->setValue(fullenergyspectrum.linelamda2);
    ui->src_aprt_59->setValue(fullenergyspectrum.naturalwidth2);
    ui->src_aprt_60->setValue(fullenergyspectrum.p2_ener);
    ui->src_aprt_61->setValue(fullenergyspectrum.linelamda3);
    ui->src_aprt_62->setValue(fullenergyspectrum.naturalwidth3);
    ui->src_aprt_64->setValue(fullenergyspectrum.p3_ener);
    ui->src_aprt_65->setValue(fullenergyspectrum.linelamda4);
    ui->src_aprt_66->setValue(fullenergyspectrum.naturalwidth4);
    ui->make_bg->setChecked(fullenergyspectrum.Do_background);
    ui->make_pol->setChecked(polarization_parameters.mka_poli);
    ui->src_aprt_46->setValue(polarization_parameters.relationP_S);

    // ================================================================================================
    // Temperature Settings
    // ================================================================================================
    ui->src_aprt_47->setValue(temperature_parameters.T_crystal_1_para);
    ui->src_aprt_48->setValue(temperature_parameters.T_crystal_2_para);
    ui->src_aprt_50->setValue(temperature_parameters.T_crystal_1_anti);
    ui->src_aprt_49->setValue(temperature_parameters.T_crystal_2_anti);
    ui->make_temp_change->setChecked(temperature_parameters.mk_temp_bin);
    ui->src_aprt_51->setValue(temperature_parameters.AA_tempera);
    ui->src_aprt_52->setValue(temperature_parameters.TT_tempera);
}