class GridOptimizer:
    def __init__(self, threshold, battery_max):
        self.threshold = threshold
        self.battery_max = battery_max
        self.battery_current = battery_max # Start full
        
    def optimize(self, predicted_load_array):
        optimized_loads = []
        battery_levels = []
        
        for load in predicted_load_array:
            load = float(load) # Ensure single float value
            
            # Logic: If Load > Threshold, Discharge Battery
            if load > self.threshold:
                discharge_needed = load - self.threshold
                discharge_amount = min(discharge_needed, self.battery_current)
                
                self.battery_current -= discharge_amount
                actual_grid_load = load - discharge_amount
                
            # Logic: If Load < Threshold, Charge Battery
            elif load < self.threshold:
                charge_room = self.battery_max - self.battery_current
                charge_amount = min(charge_room, self.threshold - load)
                
                self.battery_current += charge_amount
                actual_grid_load = load + charge_amount
                
            else:
                actual_grid_load = load
                
            optimized_loads.append(actual_grid_load)
            battery_levels.append(self.battery_current)
            
        return optimized_loads, battery_levels