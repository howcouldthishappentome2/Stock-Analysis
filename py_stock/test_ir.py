from data.stock_data import InterestRateDataCollector
import time

print('Creating InterestRateDataCollector...')
start = time.time()
collector = InterestRateDataCollector()
print(f'  Done in {time.time()-start:.1f}s')

print('Calibrating IR model...')
start = time.time()
params = collector.calibrate_ir_model()
print(f'  Done in {time.time()-start:.1f}s')

print('Getting rates...')
start = time.time()
rates = collector.get_current_rates()
print(f'  Done in {time.time()-start:.1f}s')

print(f'Success! Risk-free rate: {rates["risk_free_rate"]:.4f}')
