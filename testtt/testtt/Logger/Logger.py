import os
import csv
import numpy as np

class Logger:

    def __init__(self, filename):
        self.filename = filename[0]

        base_path = os.path.dirname(os.path.abspath(__file__))  # Get the script's directory
        base_path = os.path.join(base_path, 'data_analysis/') # Append the 'data_analysis' folder to the path

        parts = base_path.split(os.sep)        # Split the path into components
        parts = ["src" if part == "build" else part for part in parts]        # Replace 'build' with 'src' if it exists in the path
        base_path = os.sep.join(parts)        # Reconstruct the new path

        self.full_path = os.path.join(base_path, self.filename)        # Combine the base path with the filename
        print(f"Logging to: {self.full_path}")        # Print the full path
        os.makedirs(os.path.dirname(self.full_path), exist_ok=True)        # Ensure the directory exists, and creates it if it doesn't


    def log(self, ControlNode):
        with open(self.full_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['time',
                            'x', 'y', 'z', 'yaw',
                            'metadata'
                            ])
            
            time_history = ControlNode.get_time_log() #0
            x_history = ControlNode.get_x_log() #1
            y_history = ControlNode.get_y_log() #2
            z_history = ControlNode.get_z_log() #3
            yaw_history = ControlNode.get_yaw_log() #4
            ctrl_comp_time_history = ControlNode.get_ctrl_comp_time_log() #5

            # m0_history = ControlNode.get_m0_log() #6
            # m1_history = ControlNode.get_m1_log() #7
            # m2_history = ControlNode.get_m2_log() #8
            # m3_history = ControlNode.get_m3_log() #9


            # f_history = ControlNode.get_f_log() #6
            # M_history = ControlNode.get_M_log() #7

            throttle_history = ControlNode.get_throttle_log() #6
            roll_rate_history = ControlNode.get_roll_rate_log() #7
            pitch_rate_history = ControlNode.get_pitch_rate_log() #8
            yaw_rate_history = ControlNode.get_yaw_rate_log() #9

            metadata = ControlNode.get_metadata() #8 #10


            """
    def get_time_log(self): return np.array(self.time_log).reshape(-1, 1) #0
    def get_x_log(self): return np.array(self.x_log).reshape(-1, 1) #1
    def get_y_log(self): return np.array(self.y_log).reshape(-1, 1) #2
    def get_z_log(self): return np.array(self.z_log).reshape(-1, 1) #3
    def get_yaw_log(self): return np.array(self.yaw_log).reshape(-1, 1) #4
    def get_ctrl_comp_time_log(self): return np.array(self.ctrl_comp_time_log).reshape(-1, 1) #5

    -------------------------------------------------------------------
    def get_m0_log(self): return np.array(self.m0_log).reshape(-1, 1) #6
    def get_m1_log(self): return np.array(self.m1_log).reshape(-1, 1) #7
    def get_m2_log(self): return np.array(self.m2_log).reshape(-1, 1) #8
    def get_m3_log(self): return np.array(self.m3_log).reshape(-1, 1) #9
    -------------------------------------------------------------------
    def get_f_log(self): return np.array(self.f_log).reshape(-1, 1) #6
    def get_M_log(self): return np.array(self.M_log).reshape(-1, 1) #7
    -------------------------------------------------------------------
    def get_throttle_log(self): return np.array(self.throttle_log).reshape(-1, 1) #6
    def get_roll_rate_log(self): return np.array(self.roll_rate_log).reshape(-1, 1) #7
    def get_pitch_rate_log(self): return np.array(self.pitch_rate_log).reshape(-1, 1) #8
    def get_yaw_rate_log(self): return np.array(self.yaw_rate_log).reshape(-1, 1) #9

    def get_metadata(self): return self.metadata.reshape(-1, 1) #8/10
            """
            
            # Pad the metadata to match the time history
            padding_length = time_history.shape[0] - metadata.shape[0]
            metadata = np.pad(metadata, ((0, padding_length), (0, 0)), 'constant', constant_values='0')
           

            # Combine the histories for logging
            data = np.hstack((time_history,
                              x_history, y_history, z_history, yaw_history,
                              ctrl_comp_time_history,
                            #   m0_history, m1_history, m2_history, m3_history,
                            #   f_history, M_history,
                              throttle_history, roll_rate_history, pitch_rate_history, yaw_rate_history,  
                              metadata
                              ))
            # Write each row to the CSV file
            for row in range(data.shape[0]):
                writer.writerow(np.asarray(data[row, :]).flatten())

            print(f"\nWrote to {self.full_path}")