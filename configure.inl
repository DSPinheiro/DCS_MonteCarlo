#include "simuGlobals.hh"
#include <iostream>

extern double refra_corr;

struct CommandLineParams
{
    explicit CommandLineParams(bool valid) : valid(valid) {}
    CommandLineParams() : valid(true), use_rel_output(false) {}
    bool valid;

    std::string input_dir_path;
    std::string input_config_file_path;
    std::string output_dir_path;
    bool use_rel_output;
};

inline CommandLineParams command_line_parser(const int argc, char* argv[])
{
    CommandLineParams cmdlp;

    for(int i = 1; i < argc; i++)
    {
        const char* possible_desc = argv[i];

        // We have a descriptor
        // Process next parameter
        if(possible_desc[0] == '-')
        {
            switch (possible_desc[1])
            {
            case 'h':
                std::cout << "Available command line options for DCSsimu:" << std::endl;
                std::cout << "-h : Display this help message." << std::endl;
                std::cout << "-w : Specify the input configuration directory (workspace)." << std::endl;
                std::cout << "-i : Specify the input configuration filename." << std::endl;
                std::cout << "-o : Specify the ouput directory." << std::endl;
                std::cout << "-r : Use input directory workspace relative output." << std::endl;
                return CommandLineParams(false);

            case 'w':
                cmdlp.input_dir_path = std::string(argv[++i]);
                break;

            case 'i':
                cmdlp.input_config_file_path = std::string(argv[++i]);
                break;

            case 'o':
                cmdlp.output_dir_path = std::string(argv[++i]);
                break;

            case 'r':
                cmdlp.use_rel_output = true;
                break;
            
            default:
                std::cout << "Unrecognized command line option: -" << possible_desc[1] << std::endl;
                // std::cout << "Try to ignore next parameter... (This could fail)" << std::endl;
                // break;
                return CommandLineParams(false);
            }
        }
        else
        {
            std::cout << "A descriptor is required for each input in the program arguments." << std::endl;
            std::cout << "Do " << argv[0] << " -h for help." << std::endl;
            return CommandLineParams(false);
        }
    }

    return cmdlp;
}