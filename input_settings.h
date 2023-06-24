#ifndef INPUT_SETTINGS_H
#define INPUT_SETTINGS_H

#include <QWidget>

#include "simuGlobals.hh"
#include "parameter_settings.h"

QT_BEGIN_NAMESPACE
namespace Ui { class InputSettingsPrompt; }
QT_END_NAMESPACE

class InputSettingsPrompt : public QWidget
{
    Q_OBJECT

public:
    InputSettingsPrompt(GUISettingsWindow* simWindow, QWidget *parent = nullptr);
    ~InputSettingsPrompt() = default;

    static int configure(const std::string& filename);

private slots:
    void loadInputConfig();

private:
    Ui::InputSettingsPrompt* ui;
    GUISettingsWindow* simWindow;
};
#endif // INPUT_SETTINGS_H
