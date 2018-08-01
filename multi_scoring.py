import os
from joblib import Parallel, delayed
import glob
# from natsort import natsorted, ns
# Parallel(n_jobs=1)(delayed(sqrt)(i**2) for i in range(10))
# for simplicity, the folder needs to have a trailing slash
output_folder = "/data2/SNIPER_github_gpu_scoring/build_docker/result_sniper_baseline_test/"
# output_folder = "/data2/Deformable-ConvNets_submission/build_docker/result_smartchip_epoch12/"
import pathlib2
pathlib2.Path(output_folder).mkdir(parents=True, exist_ok=True) 

test_list =['7.tif', '9.tif', '11.tif', '12.tif', '27.tif', '32.tif', '35.tif', '93.tif', '96.tif', '98.tif', '108.tif', '121.tif', '122.tif', '143.tif', '147.tif', '178.tif', '201.tif', '205.tif', '207.tif', '221.tif', '239.tif', '241.tif', '259.tif', '261.tif', '297.tif', '299.tif', '301.tif', '313.tif', '319.tif', '327.tif', '332.tif', '350.tif', '354.tif', '357.tif', '358.tif', '359.tif', '365.tif', '367.tif', '368.tif', '376.tif', '396.tif', '412.tif', '420.tif', '431.tif', '436.tif', '447.tif', '483.tif', '485.tif', '490.tif', '511.tif', '530.tif', '539.tif', '543.tif', '554.tif', '564.tif', '566.tif', '576.tif', '577.tif', '578.tif', '583.tif', '586.tif', '589.tif', '592.tif', '628.tif', '631.tif', '632.tif', '636.tif', '654.tif', '656.tif', '664.tif', '666.tif', '667.tif', '668.tif', '673.tif', '676.tif', '687.tif', '690.tif', '700.tif', '710.tif', '718.tif', '731.tif', '746.tif', '754.tif', '770.tif', '780.tif', '782.tif', '809.tif', '827.tif', '882.tif', '891.tif', '908.tif', '909.tif', '912.tif', '913.tif', '925.tif', '929.tif', '941.tif', '944.tif', '945.tif', '947.tif', '949.tif', '950.tif', '961.tif', '963.tif', '965.tif', '967.tif', '978.tif', '981.tif', '985.tif', '995.tif', '1038.tif', '1040.tif', '1043.tif', '1054.tif', '1060.tif', '1062.tif', '1064.tif', '1066.tif', '1069.tif', '1071.tif', '1073.tif', '1075.tif', '1082.tif', '1097.tif', '1098.tif', '1102.tif', '1115.tif', '1116.tif', '1117.tif', '1122.tif', '1134.tif', '1138.tif', '1148.tif', '1159.tif', '1161.tif', '1177.tif', '1194.tif', '1207.tif', '1213.tif', '1234.tif', '1235.tif', '1254.tif', '1258.tif', '1263.tif', '1267.tif', '1282.tif', '1308.tif', '1326.tif', '1333.tif', '1358.tif', '1359.tif', '1360.tif', '1401.tif', '1402.tif', '1414.tif', '1426.tif', '1434.tif', '1464.tif', '1470.tif', '1471.tif', '1474.tif', '1475.tif', '1506.tif', '1510.tif', '1520.tif', '1521.tif', '1522.tif', '1554.tif', '1566.tif', '1578.tif', '1595.tif', '1599.tif', '1605.tif', '1627.tif', '1650.tif', '1652.tif', '1670.tif', '1681.tif', '1693.tif', '1728.tif', '1742.tif', '1746.tif', '1750.tif', '1765.tif', '1794.tif', '1800.tif', '1804.tif', '1811.tif', '1813.tif', '1814.tif', '1816.tif', '1819.tif', '1833.tif', '1835.tif', '1836.tif', '1843.tif', '1846.tif', '1866.tif', '1872.tif', '1882.tif', '1895.tif', '1903.tif', '1905.tif', '1909.tif', '1916.tif', '1917.tif', '1925.tif', '1933.tif', '1935.tif', '1944.tif', '1946.tif', '1951.tif', '1952.tif', '1955.tif', '1996.tif', '1997.tif', '2001.tif', '2016.tif', '2023.tif', '2024.tif', '2038.tif', '2047.tif', '2077.tif', '2100.tif', '2102.tif', '2116.tif', '2120.tif', '2127.tif', '2129.tif', '2132.tif', '2135.tif', '2146.tif', '2167.tif', '2178.tif', '2195.tif', '2300.tif', '2307.tif', '2321.tif', '2336.tif', '2340.tif', '2342.tif', '2349.tif', '2351.tif', '2357.tif', '2363.tif', '2378.tif', '2392.tif', '2393.tif', '2402.tif', '2403.tif', '2404.tif', '2411.tif', '2415.tif', '2427.tif', '2428.tif', '2429.tif', '2442.tif', '2449.tif', '2464.tif', '2473.tif', '2480.tif', '2491.tif', '2496.tif', '2500.tif', '2508.tif', '2517.tif', '2522.tif', '2529.tif', '2537.tif', '2548.tif', '2549.tif', '2554.tif', '2556.tif', '2563.tif', '2570.tif', '2575.tif', '2579.tif', '2604.tif', '2613.tif', '2620.tif', '2621.tif', '2622.tif']


gpu_index_list = [0,1,2,3]*71

#gpu_index_list = [0,0,0,0]*71
# test_list = natsorted(test_list, key=lambda y: y.lower())
# print(test_list)



def inference_single_image(input_image_name, output_folder, gpu_index):
    command = "python demo_all.py --im_path /data2/xview/val_images/" + input_image_name + " -o " + output_folder + input_image_name + ".txt --output_folder " + os.path.join("/tmp",input_image_name) 
    print(command)
    os.system(command)



# run for 3 times to make sure all the images get scored

for _ in range(3):
    # first detect if there are cache
    for filename in glob.glob(output_folder + '*.txt'):
        valid_result_name = os.path.basename(filename)[:-4]
        # print(valid_result_name)
        if valid_result_name in test_list:
            test_list.remove(valid_result_name)

    print("found incomplete evaluation:",test_list)

    Parallel(n_jobs=1)(delayed(inference_single_image)(element, output_folder, gpu_index) for element,gpu_index in zip(test_list,gpu_index_list))

