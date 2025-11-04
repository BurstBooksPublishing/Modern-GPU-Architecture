# modes definition (simple example)
set modes {\
    {name "perf"  clk_period 5.0 lib "lib_perf.db"  sdc "sdc_perf.sdc"} \
    {name "economy" clk_period 10.0 lib "lib_lp.db"   sdc "sdc_lp.sdc"} }

foreach m $modes {
    # parse map (tool-specific)
    dict with m {
        puts "Processing mode $name"
        # set timing library for this mode
        link_library $lib
        # generate SDC: create_clock and mode-specific false paths
        create_clock -name clk -period $clk_period -waveform {0 [expr {$clk_period/2}]}
        source $sdc        ;# additional false paths/input delays
        # run synthesis and write mode-tagged netlist
        compile -incremental     ;# mode-aware compile
        write_verilog ${name}_netlist.v
        write_sdf ${name}.sdf
    }
}