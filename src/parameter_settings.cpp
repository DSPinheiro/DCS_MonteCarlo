#include "../include/parameter_settings.h"
#include "../include/ui_parameter_settings.h"
#include "../include/input_settings.h"

#include "../include/simuGlobals.hh"
#include "../include/simulation_interface.h"
#include "sserializer.inl"
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <QPushButton>
#include <QCheckBox>
#include <QRadioButton>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QComboBox>
#include <QString>
#include <QFileDialog>
#include <QTimer>
#include <QMessageBox>


static std::vector<uint8_t> serialize_inputs()
{
    std::vector<uint8_t> buffer;

    serialize_data(buffer, GeometryInput);

    serialize_data(buffer, UserSettingsInput);

    serialize_data(buffer, GeoParapathlengthsInput.type_source);
    serialize_data(buffer, GeoParapathlengthsInput.LT_aper);
    serialize_data(buffer, GeoParapathlengthsInput.dist_T_Cr1);
    serialize_data(buffer, GeoParapathlengthsInput.dist_Cr1_Cr2);
    serialize_data(buffer, GeoParapathlengthsInput.dist_Cr2_Det);
    serialize_data(buffer, GeoParapathlengthsInput.dist_aper_det);

    serialize_data(buffer, GeolengthelementsInput);

    serialize_data(buffer, GeoParametersInput);

    serialize_data(buffer, CurveVerticalTiltInput);

    serialize_data(buffer, GraphOptionsInput);

    serialize_data(buffer, PlotParametersInput);

    serialize_data(buffer, NumberRaysInput);

    serialize_data(buffer, PhysicalParametersInput.Unit_energy);
    serialize_data(buffer, PhysicalParametersInput.linelamda);
    serialize_data(buffer, PhysicalParametersInput.naturalwidth);
    serialize_data(buffer, PhysicalParametersInput.gauss_Doop);

    serialize_data(buffer, PolarizationParametersInput);

    serialize_data(buffer, TemperatureParametersInput);

    // serialize_data(buffer, FullEnergySpectrumInput);
    serialize_data(buffer, FullEnergySpectrumInput.make_more_lines);
    serialize_data(buffer, FullEnergySpectrumInput.energy_spectrum_file);
    serialize_data(buffer, FullEnergySpectrumInput.linelamda1);
    serialize_data(buffer, FullEnergySpectrumInput.naturalwidth1);
    serialize_data(buffer, FullEnergySpectrumInput.p1_ener);
    serialize_data(buffer, FullEnergySpectrumInput.linelamda2);
    serialize_data(buffer, FullEnergySpectrumInput.naturalwidth2);
    serialize_data(buffer, FullEnergySpectrumInput.p2_ener);
    serialize_data(buffer, FullEnergySpectrumInput.linelamda3);
    serialize_data(buffer, FullEnergySpectrumInput.naturalwidth3);
    serialize_data(buffer, FullEnergySpectrumInput.p3_ener);
    serialize_data(buffer, FullEnergySpectrumInput.linelamda4);
    serialize_data(buffer, FullEnergySpectrumInput.naturalwidth4);
    serialize_data(buffer, FullEnergySpectrumInput.Do_background);

    serialize_data(buffer, CurvedCrystalInput);

    //TODO: UNIMPLEMENTED ANALYSIS FEATURE

    return buffer;
}

static void deserialize_inputs(std::vector<uint8_t>& buffer)
{
    consume_buffer(buffer, GeometryInput);

    consume_buffer(buffer, UserSettingsInput);

    consume_buffer(buffer, GeoParapathlengthsInput.type_source);
    consume_buffer(buffer, GeoParapathlengthsInput.LT_aper);
    consume_buffer(buffer, GeoParapathlengthsInput.dist_T_Cr1);
    consume_buffer(buffer, GeoParapathlengthsInput.dist_Cr1_Cr2);
    consume_buffer(buffer, GeoParapathlengthsInput.dist_Cr2_Det);
    consume_buffer(buffer, GeoParapathlengthsInput.dist_aper_det);

    consume_buffer(buffer, GeolengthelementsInput);

    consume_buffer(buffer, GeoParametersInput);

    consume_buffer(buffer, CurveVerticalTiltInput);

    consume_buffer(buffer, GraphOptionsInput);

    consume_buffer(buffer, PlotParametersInput);

    consume_buffer(buffer, NumberRaysInput);

    consume_buffer(buffer, PhysicalParametersInput.Unit_energy);
    consume_buffer(buffer, PhysicalParametersInput.linelamda);
    consume_buffer(buffer, PhysicalParametersInput.naturalwidth);
    consume_buffer(buffer, PhysicalParametersInput.gauss_Doop);

    consume_buffer(buffer, PolarizationParametersInput);

    consume_buffer(buffer, TemperatureParametersInput);

    // consume_buffer(buffer, FullEnergySpectrumInput);
    consume_buffer(buffer, FullEnergySpectrumInput.make_more_lines);
    consume_buffer(buffer, FullEnergySpectrumInput.energy_spectrum_file);
    consume_buffer(buffer, FullEnergySpectrumInput.linelamda1);
    consume_buffer(buffer, FullEnergySpectrumInput.naturalwidth1);
    consume_buffer(buffer, FullEnergySpectrumInput.p1_ener);
    consume_buffer(buffer, FullEnergySpectrumInput.linelamda2);
    consume_buffer(buffer, FullEnergySpectrumInput.naturalwidth2);
    consume_buffer(buffer, FullEnergySpectrumInput.p2_ener);
    consume_buffer(buffer, FullEnergySpectrumInput.linelamda3);
    consume_buffer(buffer, FullEnergySpectrumInput.naturalwidth3);
    consume_buffer(buffer, FullEnergySpectrumInput.p3_ener);
    consume_buffer(buffer, FullEnergySpectrumInput.linelamda4);
    consume_buffer(buffer, FullEnergySpectrumInput.naturalwidth4);
    consume_buffer(buffer, FullEnergySpectrumInput.Do_background);

    consume_buffer(buffer, CurvedCrystalInput);

    //TODO: UNIMPLEMENTED ANALYSIS FEATURE
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
    wsimu(new SimulationInterface()),
    timer(new QTimer(this))
{
    ui->setupUi(this);

    // Setup all of the connections to the settings

    // ================================================================================================
    // User Settings
    // ================================================================================================
    connect(ui->make_parallel, &QCheckBox::stateChanged, this, [this](int s) { UserSettingsInput.see_para = (bool)s; });
    connect(ui->make_antiparallel, &QCheckBox::stateChanged, this, [this](int s) { UserSettingsInput.see_anti = (bool)s; });
    connect(ui->vertical_divergence, &QCheckBox::stateChanged, this, [this](int s) { UserSettingsInput.Make_Vertical = (bool)s; });
    connect(ui->horizontal_divergence, &QCheckBox::stateChanged, this, [this](int s) { UserSettingsInput.Make_Horizontal = (bool)s; });
    connect(ui->fitting, &QCheckBox::stateChanged, this, [this](int s) { UserSettingsInput.fitting = (bool)s; });
    connect(ui->fitting_truev, &QCheckBox::stateChanged, this, [this](int s) { UserSettingsInput.TrueVoigt = (bool)s; });
    connect(ui->center_c1, &QCheckBox::stateChanged, this, [this](int s) { UserSettingsInput.center_1crys = (bool)s; });
    connect(ui->center_c2, &QCheckBox::stateChanged, this, [this](int s) { UserSettingsInput.center_2crys = (bool)s; });
    connect(ui->print_scan, &QCheckBox::stateChanged, this, [this](int s) { UserSettingsInput.print_scan = (bool)s; });
    connect(ui->center_mask, &QCheckBox::stateChanged, this, [this](int s) { UserSettingsInput.center_Mask = (bool)s; });
    connect(ui->test_mask, &QCheckBox::stateChanged, this, [this](int s) { UserSettingsInput.make_mask_test = (bool)s; });
    connect(ui->aa_none, &QRadioButton::clicked, this, [this]() { UserSettingsInput.angle_aprox = 0; });
    connect(ui->aa_partial, &QRadioButton::clicked, this, [this]() { UserSettingsInput.angle_aprox = 1; });
    connect(ui->aa_full, &QRadioButton::clicked, this, [this]() { UserSettingsInput.angle_aprox = 2; });
    connect(ui->c1m_no, &QRadioButton::clicked, this, [this]() { UserSettingsInput.mask_C1 = 0; });
    connect(ui->c1m_down, &QRadioButton::clicked, this, [this]() { UserSettingsInput.mask_C1 = 1; });
    connect(ui->c1m_up, &QRadioButton::clicked, this, [this]() { UserSettingsInput.mask_C1 = 2; });
    connect(ui->c2m_no, &QRadioButton::clicked, this, [this]() { UserSettingsInput.mask_C2 = 0; });
    connect(ui->c2m_down, &QRadioButton::clicked, this, [this]() { UserSettingsInput.mask_C2 = 1; });
    connect(ui->c2m_up, &QRadioButton::clicked, this, [this]() { UserSettingsInput.mask_C2 = 2; });
    connect(ui->st_simple, &QRadioButton::clicked, this, [this]() { UserSettingsInput.Simple_simu = true; });
    connect(ui->st_complex, &QRadioButton::clicked, this, [this]() { UserSettingsInput.Simple_simu = false; });

    // ================================================================================================
    // Geometric Settings
    // ================================================================================================
    connect(ui->spinBox_5, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int v) { GeometryInput.imh = v; });
    connect(ui->spinBox_6, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int v) { GeometryInput.imk = v; });
    connect(ui->spinBox_7, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int v) { GeometryInput.iml = v; });
    // Geometry mode (reserved)
    connect(ui->src_geometry_6, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int v) { 
        switch(v)
        {
            case 0: GeometryInput.crystal_Si = true; break;
            case 1: GeometryInput.crystal_Si = false; break;
            default: break;
        }
    });
    connect(ui->src_geometry, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int v) { 
        switch(v)
        {
            case 0: GeoParapathlengthsInput.type_source = "UC"; break;
            case 1: GeoParapathlengthsInput.type_source = "UR"; break;
            case 2: GeoParapathlengthsInput.type_source =  "P"; break;
            case 3: GeoParapathlengthsInput.type_source =  "G"; break;
            default: break;
        }
    });
    connect(ui->src_aprt_11, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { GeolengthelementsInput.S_sour = v; });
    connect(ui->src_aprt_12, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { GeolengthelementsInput.S_aper = v; });
    connect(ui->src_aprt_18, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { GeolengthelementsInput.y_sour = v; });
    connect(ui->src_aprt_19, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { GeolengthelementsInput.z_sour = v; });
    connect(ui->src_aprt_20, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { GeolengthelementsInput.y_aper = v; });
    connect(ui->src_aprt_21, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { GeolengthelementsInput.z_aper = v; });
    connect(ui->src_aprt_13, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { GeolengthelementsInput.S_aper_var = v; });
    connect(ui->src_aprt_14, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { GeolengthelementsInput.S_shi_hor_A = v; });
    connect(ui->src_aprt_15, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { GeolengthelementsInput.S_shi_ver_A = v; });
    connect(ui->src_aprt_16, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { GeolengthelementsInput.S_shi_hor_B = v; });
    connect(ui->src_aprt_17, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { GeolengthelementsInput.S_shi_ver_B = v; });
    connect(ui->src_aprt   , QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { GeoParapathlengthsInput.LT_aper = v; });
    connect(ui->src_aprt_2 , QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { GeoParapathlengthsInput.dist_T_Cr1 = v; });
    connect(ui->src_aprt_3 , QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { GeoParapathlengthsInput.dist_Cr1_Cr2 = v; });
    connect(ui->src_aprt_4 , QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { GeoParapathlengthsInput.dist_Cr2_Det = v; });
    connect(ui->src_aprt_6 , QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { GeolengthelementsInput.y_first_crys = v; });
    connect(ui->src_aprt_7 , QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { GeolengthelementsInput.z_first_crys = v; });
    connect(ui->src_aprt_8 , QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { GeolengthelementsInput.ydetc = v; });
    connect(ui->src_aprt_9 , QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { GeolengthelementsInput.zdetc = v; });
    connect(ui->src_aprt_10, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { GeolengthelementsInput.shift_det_ver = v; });

    // ================================================================================================
    // Crystal Settings
    // ================================================================================================
    connect(ui->src_aprt_22, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { GeoParametersInput.Exp_crys1 = v; });
    connect(ui->src_aprt_23, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { GeoParametersInput.teta_table = v; });
    connect(ui->src_aprt_24, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { GeoParametersInput.OffsetRotCry1 = v; });
    connect(ui->src_aprt_25, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { GeoParametersInput.teta_detec_para = v; });
    connect(ui->src_aprt_26, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { GeoParametersInput.teta_detec_anti = v; });
    connect(ui->src_aprt_27, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { GeoParametersInput.tilt_C1 = v; });
    connect(ui->src_aprt_29, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { GeoParametersInput.tilt_C2 = v; });
    connect(ui->src_aprt_30, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { GeoParametersInput.xsi = v; });
    connect(ui->src_aprt_31, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { GeoParametersInput.center_1cry_at = v; });
    connect(ui->src_aprt_32, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { GeoParametersInput.center_2cry_at = v; });
    connect(ui->make_img_plates_7, &QCheckBox::stateChanged, this, [this](int s) { GeoParametersInput.make_table_noise = (bool)s; });
    connect(ui->src_aprt_39, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { GeoParametersInput.table_resolution = v; });
    connect(ui->src_aprt_33, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { CurveVerticalTiltInput.phas_tilt1 = v; });
    connect(ui->src_aprt_34, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { CurveVerticalTiltInput.phas_tilt2 = v; });
    connect(ui->src_aprt_35, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { CurveVerticalTiltInput.offsettilt1 = v; });
    connect(ui->src_aprt_36, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { CurveVerticalTiltInput.offsettilt2 = v; });
    connect(ui->src_aprt_37, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { CurveVerticalTiltInput.consttilt1 = v; });
    connect(ui->src_aprt_38, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { CurveVerticalTiltInput.consttilt2 = v; });
    connect(ui->make_img_plates_6, &QCheckBox::stateChanged, this, [this](int s) { CurvedCrystalInput.Curve_crystall = (bool)s; });
    connect(ui->src_aprt_53, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { CurvedCrystalInput.R_cur_crys_1 = v; });
    connect(ui->src_aprt_54, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { CurvedCrystalInput.R_cur_crys_2 = v; });

    // ================================================================================================
    // Plot Settings
    // ================================================================================================
    connect(ui->make_img_plates_5, &QCheckBox::stateChanged, this, [this](int s) { GraphOptionsInput.make_graph_profile = (bool)s; });
    connect(ui->make_img_plates  , &QCheckBox::stateChanged, this, [this](int s) { GraphOptionsInput.make_image_plates = (bool)s; });
    connect(ui->make_img_plates_3, &QCheckBox::stateChanged, this, [this](int s) { GraphOptionsInput.make_imageC1_After_refle = (bool)s; });
    connect(ui->make_img_plates_4, &QCheckBox::stateChanged, this, [this](int s) { GraphOptionsInput.make_imageC2_After_refle = (bool)s; });
    connect(ui->src_aprt_40, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { PlotParametersInput.delta_angl = v; });
    connect(ui->src_aprt_41, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { PlotParametersInput.shift_disp_window = v; });
    connect(ui->spinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int v) { PlotParametersInput.nubins = v; });

    // ================================================================================================
    // Other Settings
    // ================================================================================================
    connect(ui->spinBox_2, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int v) { NumberRaysInput.nbeams = v; });
    connect(ui->spinBox_3, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int v) { NumberRaysInput.number_rotati = v; });
    connect(ui->spinBox_4, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int v) { NumberRaysInput.number_events = v; });
    connect(ui->src_geometry_3, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int v) { 
        switch(v)
        {
            case 0: PhysicalParametersInput.Unit_energy =  "eV"; break;
            case 1: PhysicalParametersInput.Unit_energy = "keV"; break;
            case 2: PhysicalParametersInput.Unit_energy =   "A"; break;
            default: break;
        }
    });
    connect(ui->src_aprt_42, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { PhysicalParametersInput.linelamda = v; });
    connect(ui->src_aprt_43, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { PhysicalParametersInput.naturalwidth = v; });
    connect(ui->src_aprt_45, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { PhysicalParametersInput.gauss_Doop = v; });
    connect(ui->src_geometry_4, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int v) { 
        switch(v)
        {
            case 0:
                FullEnergySpectrumInput.make_more_lines = 2;
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
                FullEnergySpectrumInput.make_more_lines = 1;
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
                FullEnergySpectrumInput.make_more_lines = 0;
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

        FullEnergySpectrumInput.energy_spectrum_file = filename.toStdString();
        ui->lineEdit->setText(QString::fromStdString(FullEnergySpectrumInput.energy_spectrum_file));
    });

    connect(ui->src_aprt_55, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { FullEnergySpectrumInput.linelamda1 = v; });
    connect(ui->src_aprt_56, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { FullEnergySpectrumInput.naturalwidth1 = v; });
    connect(ui->src_aprt_57, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { FullEnergySpectrumInput.p1_ener = v; });
    connect(ui->src_aprt_58, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { FullEnergySpectrumInput.linelamda2 = v; });
    connect(ui->src_aprt_59, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { FullEnergySpectrumInput.naturalwidth2 = v; });
    connect(ui->src_aprt_60, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { FullEnergySpectrumInput.p2_ener = v; });
    connect(ui->src_aprt_61, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { FullEnergySpectrumInput.linelamda3 = v; });
    connect(ui->src_aprt_62, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { FullEnergySpectrumInput.naturalwidth3 = v; });
    connect(ui->src_aprt_64, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { FullEnergySpectrumInput.p3_ener = v; });
    connect(ui->src_aprt_65, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { FullEnergySpectrumInput.linelamda4 = v; });
    connect(ui->src_aprt_66, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { FullEnergySpectrumInput.naturalwidth4 = v; });
    connect(ui->make_bg, &QCheckBox::stateChanged, this, [this](int s) { FullEnergySpectrumInput.Do_background = (bool)s; });
    
    connect(ui->make_pol, &QCheckBox::stateChanged, this, [this](int s) { PolarizationParametersInput.mka_poli = (bool)s; });
    connect(ui->src_aprt_46, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { PolarizationParametersInput.relationP_S = v; });

    connect(ui->make_gpu, &QCheckBox::stateChanged, this, [this](int s) { ParallelSettingsInput.Make_GPU = (bool)s; ui->src_aprt_63->setEnabled(!s); });
    connect(ui->src_aprt_63, QOverload<int>::of(&QSpinBox::valueChanged), this, [this](int v) { ParallelSettingsInput.OMP_threads = v; });

    // ================================================================================================
    // Temperature Settings
    // ================================================================================================
    connect(ui->src_aprt_47, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { TemperatureParametersInput.T_crystal_1_para = v; });
    connect(ui->src_aprt_48, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { TemperatureParametersInput.T_crystal_2_para = v; });
    connect(ui->src_aprt_50, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { TemperatureParametersInput.T_crystal_1_anti = v; });
    connect(ui->src_aprt_49, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { TemperatureParametersInput.T_crystal_2_anti = v; });
    connect(ui->make_temp_change, &QCheckBox::stateChanged, this, [this](int s) { TemperatureParametersInput.mk_temp_bin = (bool)s; });
    connect(ui->src_aprt_51, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { TemperatureParametersInput.AA_tempera = v; });
    connect(ui->src_aprt_52, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double v) { TemperatureParametersInput.TT_tempera = v; });

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
            tr("DCSsimu params (*.dsp);;DCSimu input file (*.input)")
        );

        if(filename.isEmpty())
        {
            return;
        }

        if(filename.endsWith(".input"))
        {
            InputSettingsPrompt::configure(filename.toStdString());
        }
        else
        {
            auto buffer = read_buffer_from_file(filename.toStdString());
            deserialize_inputs(buffer);
        }

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
    connect(ui->startSim_button, &QPushButton::clicked, wsimu, [this]() {
        struct stat buffer;
        bool exist_file = (stat(FullEnergySpectrumInput.energy_spectrum_file.c_str(), &buffer) == 0);

        if (!exist_file) {
            energyFileMissingDialog();
            wsimu->setPctDone(1.0);
        }
        else
        {
            wsimu->startSimulationThread();
        }
    });
    ui->startSim_button->setEnabled(false);
    ui->startSim_button->setStyleSheet(QString(
        "border-color: rgb(0, 0, 0);"
        "background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0,stop:0 rgba(100, 100, 100, 255), stop:1 rgba(100, 100, 100, 255));"
    ));
    connect(ui->startSim_button, &QPushButton::clicked, this, [this]() { this->setEnabled(false); });
#else
    connect(ui->startSim_button, &QPushButton::clicked, wsimu, [this]() {
        struct stat buffer;
        bool exist_file = (stat(FullEnergySpectrumInput.energy_spectrum_file.c_str(), &buffer) == 0);

        if (!exist_file) {
            energyFileMissingDialog();
            wsimu->setPctDone(1.0);
        }
        else
        {
            wsimu->show();
        }
    });
    ui->startSim_button->setEnabled(true);
    ui->startSim_button->setStyleSheet(QString(
        "border-color: rgb(0, 0, 0);"
        "background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0,stop:0 rgba(0, 100, 0, 255), stop:1 rgba(0, 170, 0, 255));"
    ));
#endif

    connect(ui->startSim_button, &QPushButton::clicked, this, [this]() { timer->start(100); });
    connect(timer, &QTimer::timeout, this, [this]() {
        double pct = wsimu->getPctDone();
        ui->startSim_button->setStyleSheet(QString(
            "border-color: rgb(0, 0, 0);"
            "background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0,"
            "stop:0 rgba(0, 100, 0, 255), stop:%1 rgba(0, 170, 0, 255),"
            "stop:%2 rgba(100, 100, 100, 255), stop:1 rgba(100, 100, 100, 255));"
        ).arg(std::max(pct-0.02, 0.0)).arg(pct));
        int ipct = static_cast<int>(pct * 100.0);
        ui->startSim_button->setText(QString::number(ipct) + "%");

        if(pct >= 0.999)
        {
            timer->stop();
            ui->startSim_button->setStyleSheet(QString(
                "border-color: rgb(0, 0, 0);"
                "background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0,stop:0 rgba(0, 100, 0, 255), stop:1 rgba(0, 170, 0, 255));"
            ));
            ui->startSim_button->setText("Start Simulation");
        }
    });
#ifdef LIB_DEF
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
    Unit_energy = PhysicalParametersInput.Unit_energy;
    if(GeometryInput.imh == 2 and GeometryInput.imk == 2 and GeometryInput.iml == 2)
    {
        refra_corr = refra_corrNIST;
    }
    else
    {
        refra_corr = refra_corrPARIS;
    }
#endif
}

GUISettingsWindow::~GUISettingsWindow()
{
    delete ui;
}

void GUISettingsWindow::energyFileMissingDialog()
{
    QString message = "Could not open energy spectrum file: ";
    message.append(QString::fromUtf8(FullEnergySpectrumInput.energy_spectrum_file.c_str()));

    QMessageBox msgBox;
    msgBox.setText(message);
    msgBox.setInformativeText("Please check if the file still exists or has been corrupted.");
    msgBox.setStandardButtons(QMessageBox::Close);
    msgBox.setDefaultButton(QMessageBox::Close);
    int ret = msgBox.exec();
}

void GUISettingsWindow::setup()
{
    ui->lineEdit->setPlaceholderText(QString(File_simu) + "Energy_spectrum.txt");
    FullEnergySpectrumInput.energy_spectrum_file = ui->lineEdit->placeholderText().toStdString();

    #ifndef OPENMP
        ui->src_aprt_63->setEnabled(false);
    #else
        ui->src_aprt_63->setMaximum(ParallelSettingsInput.system_max_threads);
    #endif
    
    #ifndef CUDA
        ui->make_gpu->setEnabled(false);
    #endif
}

void GUISettingsWindow::updateElements()
{
    // ================================================================================================
    // User Settings
    // ================================================================================================
    ui->make_parallel->setChecked(UserSettingsInput.see_para);
    ui->make_antiparallel->setChecked(UserSettingsInput.see_anti);
    ui->vertical_divergence->setChecked(UserSettingsInput.Make_Vertical);
    ui->horizontal_divergence->setChecked(UserSettingsInput.Make_Horizontal);
    ui->fitting->setChecked(UserSettingsInput.fitting);
    ui->fitting_truev->setChecked(UserSettingsInput.TrueVoigt);
    ui->center_c1->setChecked(UserSettingsInput.center_1crys);
    ui->center_c2->setChecked(UserSettingsInput.center_2crys);
    ui->print_scan->setChecked(UserSettingsInput.print_scan);
    ui->center_mask->setChecked(UserSettingsInput.center_Mask);
    ui->test_mask->setChecked(UserSettingsInput.make_mask_test);
    ui->aa_none->setChecked(UserSettingsInput.angle_aprox == 0);
    ui->aa_partial->setChecked(UserSettingsInput.angle_aprox == 1);
    ui->aa_full->setChecked(UserSettingsInput.angle_aprox == 2);
    ui->c1m_no->setChecked(UserSettingsInput.mask_C1 == 0);
    ui->c1m_down->setChecked(UserSettingsInput.mask_C1 == 1);
    ui->c1m_up->setChecked(UserSettingsInput.mask_C1 == 2);
    ui->c2m_no->setChecked(UserSettingsInput.mask_C2 == 0);
    ui->c2m_down->setChecked(UserSettingsInput.mask_C2 == 1);
    ui->c2m_up->setChecked(UserSettingsInput.mask_C2 == 2);
    ui->st_simple->setChecked(UserSettingsInput.Simple_simu);
    ui->st_complex->setChecked(!UserSettingsInput.Simple_simu);

    // ================================================================================================
    // Geometric Settings
    // ================================================================================================
    ui->spinBox_5->setValue(GeometryInput.imh);
    ui->spinBox_6->setValue(GeometryInput.imk);
    ui->spinBox_7->setValue(GeometryInput.iml);
    // Geometry mode (reserved)
    if(GeometryInput.crystal_Si)      ui->src_geometry_6->setCurrentIndex(0);
    else if(!GeometryInput.crystal_Si) ui->src_geometry_6->setCurrentIndex(1);
    if(GeoParapathlengthsInput.type_source == "UC")      ui->src_geometry->setCurrentIndex(0);
    else if(GeoParapathlengthsInput.type_source == "UR") ui->src_geometry->setCurrentIndex(1);
    else if(GeoParapathlengthsInput.type_source ==  "P") ui->src_geometry->setCurrentIndex(2);
    else if(GeoParapathlengthsInput.type_source ==  "G") ui->src_geometry->setCurrentIndex(3);
    ui->src_aprt_11->setValue(GeolengthelementsInput.S_sour);
    ui->src_aprt_12->setValue(GeolengthelementsInput.S_aper);
    ui->src_aprt_18->setValue(GeolengthelementsInput.y_sour);
    ui->src_aprt_19->setValue(GeolengthelementsInput.z_sour);
    ui->src_aprt_20->setValue(GeolengthelementsInput.y_aper);
    ui->src_aprt_21->setValue(GeolengthelementsInput.z_aper);
    ui->src_aprt_13->setValue(GeolengthelementsInput.S_aper_var);
    ui->src_aprt_14->setValue(GeolengthelementsInput.S_shi_hor_A);
    ui->src_aprt_15->setValue(GeolengthelementsInput.S_shi_ver_A);
    ui->src_aprt_16->setValue(GeolengthelementsInput.S_shi_hor_B);
    ui->src_aprt_17->setValue(GeolengthelementsInput.S_shi_ver_B);
    ui->src_aprt->setValue(GeoParapathlengthsInput.LT_aper);
    ui->src_aprt_2->setValue(GeoParapathlengthsInput.dist_T_Cr1);
    ui->src_aprt_3->setValue(GeoParapathlengthsInput.dist_Cr1_Cr2);
    ui->src_aprt_4->setValue(GeoParapathlengthsInput.dist_Cr2_Det);
    ui->src_aprt_6->setValue(GeolengthelementsInput.y_first_crys);
    ui->src_aprt_7->setValue(GeolengthelementsInput.z_first_crys);
    ui->src_aprt_8->setValue(GeolengthelementsInput.ydetc);
    ui->src_aprt_9->setValue(GeolengthelementsInput.zdetc);
    ui->src_aprt_10->setValue(GeolengthelementsInput.shift_det_ver);

    // ================================================================================================
    // Crystal Settings
    // ================================================================================================
    ui->src_aprt_22->setValue(GeoParametersInput.Exp_crys1);
    ui->src_aprt_23->setValue(GeoParametersInput.teta_table);
    ui->src_aprt_24->setValue(GeoParametersInput.OffsetRotCry1);
    ui->src_aprt_25->setValue(GeoParametersInput.teta_detec_para);
    ui->src_aprt_26->setValue(GeoParametersInput.teta_detec_anti);
    ui->src_aprt_27->setValue(GeoParametersInput.tilt_C1);
    ui->src_aprt_29->setValue(GeoParametersInput.tilt_C2);
    ui->src_aprt_30->setValue(GeoParametersInput.xsi);
    ui->src_aprt_31->setValue(GeoParametersInput.center_1cry_at);
    ui->src_aprt_32->setValue(GeoParametersInput.center_2cry_at);
    ui->src_aprt_39->setValue(GeoParametersInput.table_resolution);
    ui->make_img_plates_7->setChecked(GeoParametersInput.make_table_noise);
    ui->src_aprt_33->setValue(CurveVerticalTiltInput.phas_tilt1);
    ui->src_aprt_34->setValue(CurveVerticalTiltInput.phas_tilt2);
    ui->src_aprt_35->setValue(CurveVerticalTiltInput.offsettilt1);
    ui->src_aprt_36->setValue(CurveVerticalTiltInput.offsettilt2);
    ui->src_aprt_37->setValue(CurveVerticalTiltInput.consttilt1);
    ui->src_aprt_38->setValue(CurveVerticalTiltInput.consttilt2);
    ui->make_img_plates_6->setChecked(CurvedCrystalInput.Curve_crystall);
    ui->src_aprt_53->setValue(CurvedCrystalInput.R_cur_crys_1);
    ui->src_aprt_54->setValue(CurvedCrystalInput.R_cur_crys_2);

    // ================================================================================================
    // Plot Settings
    // ================================================================================================
    ui->make_img_plates_5->setChecked(GraphOptionsInput.make_graph_profile);
    ui->make_img_plates->setChecked(GraphOptionsInput.make_image_plates);
    ui->make_img_plates_3->setChecked(GraphOptionsInput.make_imageC1_After_refle);
    ui->make_img_plates_4->setChecked(GraphOptionsInput.make_imageC2_After_refle);
    ui->src_aprt_40->setValue(PlotParametersInput.delta_angl);
    ui->src_aprt_41->setValue(PlotParametersInput.shift_disp_window);
    ui->spinBox->setValue(PlotParametersInput.nubins);

    // ================================================================================================
    // Other Settings
    // ================================================================================================
    ui->spinBox_2->setValue(NumberRaysInput.nbeams);
    ui->spinBox_3->setValue(NumberRaysInput.number_rotati);
    ui->spinBox_4->setValue(NumberRaysInput.number_events);
    if(PhysicalParametersInput.Unit_energy ==  "eV")      ui->src_geometry_3->setCurrentIndex(0);
    else if(PhysicalParametersInput.Unit_energy == "keV") ui->src_geometry_3->setCurrentIndex(1);
    else if(PhysicalParametersInput.Unit_energy ==   "A") ui->src_geometry_3->setCurrentIndex(2);
    ui->src_aprt_42->setValue(PhysicalParametersInput.linelamda);
    ui->src_aprt_43->setValue(PhysicalParametersInput.naturalwidth);
    ui->src_aprt_45->setValue(PhysicalParametersInput.gauss_Doop);
    switch(FullEnergySpectrumInput.make_more_lines)
    {
        case 2: ui->src_geometry_4->setCurrentIndex(0); emit ui->src_geometry_4->currentIndexChanged(0); break;
        case 1: ui->src_geometry_4->setCurrentIndex(1); emit ui->src_geometry_4->currentIndexChanged(0); break;
        case 0: ui->src_geometry_4->setCurrentIndex(2); emit ui->src_geometry_4->currentIndexChanged(0); break;
        default: break;
    }
    ui->lineEdit->setText(QString::fromStdString(FullEnergySpectrumInput.energy_spectrum_file));
    ui->src_aprt_55->setValue(FullEnergySpectrumInput.linelamda1);
    ui->src_aprt_56->setValue(FullEnergySpectrumInput.naturalwidth1);
    ui->src_aprt_57->setValue(FullEnergySpectrumInput.p1_ener);
    ui->src_aprt_58->setValue(FullEnergySpectrumInput.linelamda2);
    ui->src_aprt_59->setValue(FullEnergySpectrumInput.naturalwidth2);
    ui->src_aprt_60->setValue(FullEnergySpectrumInput.p2_ener);
    ui->src_aprt_61->setValue(FullEnergySpectrumInput.linelamda3);
    ui->src_aprt_62->setValue(FullEnergySpectrumInput.naturalwidth3);
    ui->src_aprt_64->setValue(FullEnergySpectrumInput.p3_ener);
    ui->src_aprt_65->setValue(FullEnergySpectrumInput.linelamda4);
    ui->src_aprt_66->setValue(FullEnergySpectrumInput.naturalwidth4);
    ui->make_bg->setChecked(FullEnergySpectrumInput.Do_background);
    ui->make_gpu->setChecked(ParallelSettingsInput.Make_GPU);
    ui->src_aprt_63->setEnabled(!ParallelSettingsInput.Make_GPU);
    ui->src_aprt_63->setValue(ParallelSettingsInput.OMP_threads);
    ui->make_pol->setChecked(PolarizationParametersInput.mka_poli);
    ui->src_aprt_46->setValue(PolarizationParametersInput.relationP_S);

    // ================================================================================================
    // Temperature Settings
    // ================================================================================================
    ui->src_aprt_47->setValue(TemperatureParametersInput.T_crystal_1_para);
    ui->src_aprt_48->setValue(TemperatureParametersInput.T_crystal_2_para);
    ui->src_aprt_50->setValue(TemperatureParametersInput.T_crystal_1_anti);
    ui->src_aprt_49->setValue(TemperatureParametersInput.T_crystal_2_anti);
    ui->make_temp_change->setChecked(TemperatureParametersInput.mk_temp_bin);
    ui->src_aprt_51->setValue(TemperatureParametersInput.AA_tempera);
    ui->src_aprt_52->setValue(TemperatureParametersInput.TT_tempera);
}