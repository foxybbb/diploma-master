% ===========================================================================
% Copyright (C) 2021-2023 Infineon Technologies AG
%
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:
%
% 1. Redistributions of source code must retain the above copyright notice,
%    this list of conditions and the following disclaimer.
% 2. Redistributions in binary form must reproduce the above copyright
%    notice, this list of conditions and the following disclaimer in the
%    documentation and/or other materials provided with the distribution.
% 3. Neither the name of the copyright holder nor the names of its
%    contributors may be used to endorse or promote products derived from
%    this software without specific prior written permission.
%
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
% POSSIBILITY OF SUCH DAMAGE.
% ===========================================================================

addpath('lib');
addpath('..\common');

if(exist('Dev','var'))
    clear Dev;
end

%##########################################################################
% Check for library dependancies
%##########################################################################
try
    MimoseDevice.check_path();
catch
    ME = MException(['RadarPath:error'], 'ATR22 MexWrapper not found, add path to MimoseSDKMEXWrapper');
    throw(ME);
end
try
    disp(['Radar SDK Version: ' MimoseDevice.get_version_full()]);
catch
    ME = MException(['RadarDevice:error'], 'SDK library not found, add path containing compiled ATR22 Mex file and RDK libraries');
    throw(ME);
end
%##########################################################################
% STEP 1: Create a Mimose device object and connect to an attached device
%##########################################################################
Dev = MimoseDevice();

%##########################################################################
% STEP 2: Modify configuration parameters.
%##########################################################################
% Modify default configuration
frame_repetition_time = 0.05; %20 Hz
Dev.config.FrameConfig{1}.num_samples = 32;
Dev.config.FrameConfig{1}.selected_pulse_config_1 = true;
Dev.config.PulseConfig{1}.tx_power_level = 60;
Dev.config.PulseConfig{2}.tx_power_level = 63;
Dev.config.FrameConfig{1}.frame_repetition_time_s = frame_repetition_time;
Dev.config.FrameConfig{1}.pulse_repetition_time_s = 0.0012;

%##########################################################################
% STEP 3:  Configure the Radar device using the DeviceConfig object updated
% in the last STEP.
%##########################################################################
Dev.set_config(); %sends the device config to the device
configuration = Dev.config;

%##########################################################################
% STEP 4:  If RC clock is enabled and the returned system clock deviates
% more than the desired clock then the RC look up table can be tuned
%##########################################################################
%Dev.update_rc_lut();

%##########################################################################
% Algorithm parameters
%##########################################################################
fft_size = 128;
power_threshold = -50;

%##########################################################################
% create the Algorithm and plotting objects
%##########################################################################
plot_obj = demo_gui();
process_obj = frame_process('fft_size', fft_size, 'num_samples', configuration.FrameConfig{1}.num_samples);
kd_object = kick_detection('power_threshold', power_threshold);

countdown = 0;
mode = plot_obj.get_sto_mode();
while ishandle(plot_obj.hfig)
    % Fetch next frame data from the RadarDevice
    last_frame = Dev.get_next_frame();
    if (isempty(last_frame))
        continue;
    end
    % Do some processing with the obtained frame.
    output = process_obj.run(last_frame);

    [ kd_object, kick_detected]  = kd_object.run(output);
    plot_obj = plot_obj.update_plot(output, power_threshold);

    if(kick_detected)
        Dev.stop_acquisition();
        disp('................VALID KICK');
        plot_obj = plot_obj.kick_detected();
    end

    % sniffing mode simulation
    if(ishandle(plot_obj.hfig))
        new_mode = plot_obj.get_sto_mode();
    end
    if(~isequal(new_mode, mode))
        if(new_mode==0)
            Dev.stop_acquisition();
            Dev.config.FrameConfig{1}.frame_repetition_time_s = frame_repetition_time*10;
            Dev.set_config();
        else
            Dev.stop_acquisition();
            Dev.config.FrameConfig{1}.frame_repetition_time_s = frame_repetition_time;
            Dev.set_config();
        end
        mode = new_mode;
    end
end

%##########################################################################
% STEP 6: Stop the Radar data acquisition trigerred in the last step by the
% function get_next_frame(). Now the device can be reconfigured and
% re-triggered by get_next_frame()
%##########################################################################
Dev.stop_acquisition();

%##########################################################################
% STEP 7: Clear the RadarDevice object. It also automatically disconnects
% from the device.
%##########################################################################
clear Dev
