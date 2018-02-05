from pathnet_keras import PathNet
from path_search import PathSearch, TS_box
from dataprep import DataPrep
from plot_pathnet import PathNetPlotter


training_size = 60000
test_size = 5000

data = DataPrep()
data.mnist(train_size=training_size, test_size=test_size)
data.add_padding()
data.grayscale2rgb()



x1, y1, x_test1, y_test1 = data.sample_dataset([0, 1, 2, 3, 4])
x2, y2, x_test2, y_test2 = data.sample_dataset([5, 6, 7, 8, 9])
x3, y3, x_test3, y_test3 = data.sample_dataset([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
data = DataPrep()
data.cSVHN_ez()
data.x = data.x[:training_size]
data.y = data.y[:training_size]
data.x_test = data.x_test[:test_size]
data.y_test = data.y_test[:test_size]

x4, y4, x_test4, y_test4 = data.sample_dataset([0, 1, 2, 3, 4])
x5, y5, x_test5, y_test5 = data.sample_dataset([5, 6, 7, 8, 9])
x6, y6, x_test6, y_test6 = data.sample_dataset([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
i = 0
while True:
    i+=1
    pn, task1 = PathNet.mnist(output_size=5, image_shape=data.x[0].shape)
    ps = PathSearch(pn)

    hyperparam = {'threshold_acc': 0.975,
                  'generation_limit': 500,
                  'population_size': 24,
                  'selection_preassure': 3,
                  'replace_func': TS_box.replace_3to2}

    hyperparam['name'] = 'Task 1: MNIST [0, 1, 2, 3, 4]'
    path1, fitness1, log1 = ps.new_tournamet_search(x1, y1, task1, hyperparam=hyperparam)
    pn.save_new_optimal_path(path1, task1)
    print('Training accuracy:', fitness1)
    print('Test accuracy: ', pn.evaluate_path(x_test1, y_test1, path1, task=task1))


    hyperparam['name'] = 'Task 2: MNIST [5, 6, 7, 8, 9]'
    task2 = pn.create_new_task(like_this=task1)
    task2.name = 'unique_2'
    path2, fitness2, log2 = ps.new_tournamet_search(x2, y2, task2, hyperparam=hyperparam)
    pn.save_new_optimal_path(path2, task2)
    print('Training accuracy:', fitness2)
    print('Test accuracy: ', pn.evaluate_path(x_test2, y_test2, path2, task=task2))

    hyperparam['threshold_acc'] = 0.85

    hyperparam['name'] = 'Task 3: MNIST [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]'
    task3_config = task2.get_defining_config()
    task3_config['name'] = 'unique_3'
    task3_config['output'] = 10
    task3 = pn.create_new_task(config=task3_config)
    path3, fitness3, log3 = ps.new_tournamet_search(x3, y3, task3, hyperparam=hyperparam)
    pn.save_new_optimal_path(path3, task3)
    print('Training accuracy:', fitness3)
    print('Test accuracy: ', pn.evaluate_path(x_test3, y_test3, path3, task=task3))


    hyperparam['name'] = 'Task 4: cSVHN [0, 1, 2, 3, 4]'
    task4 = pn.create_new_task(like_this=task2)
    task4.name = 'unique_4'
    path4, fitness4, log4 = ps.new_tournamet_search(x4, y4, task4, hyperparam=hyperparam)
    pn.save_new_optimal_path(path4, task4)
    print('Training accuracy:', fitness4)
    print('Test accuracy: ', pn.evaluate_path(x_test4, y_test4, path4, task=task4))

    hyperparam['name'] = 'Task 5: cSVHN [5, 6, 7, 8, 9]'
    task5 = pn.create_new_task(like_this=task4)
    task5.name = 'unique_5'
    path5, fitness5, log5 = ps.new_tournamet_search(x5, y5, task5, hyperparam=hyperparam)
    pn.save_new_optimal_path(path5, task5)
    print('Training accuracy:', fitness5)
    print('Test accuracy: ', pn.evaluate_path(x_test5, y_test5, path5, task=task5))

    hyperparam['name'] = 'Task 6: cSVHN [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]'
    task6_config = task5.get_defining_config()
    task6_config['name'] = 'unique_6'
    task6_config['output'] = 10
    task6 = pn.create_new_task(config=task6_config)
    path6, fitness6, log6 = ps.new_tournamet_search(x6, y6, task6, hyperparam=hyperparam)
    pn.save_new_optimal_path(path6, task6)
    print('Training accuracy:', fitness6)
    print('Test accuracy: ', pn.evaluate_path(x_test6, y_test6, path6, task=task6))


    #print('Overlap MNIST -> cSVHN same classes[A->A]:', Analytic.path_overlap(path1, path4))
    #print('Overlap MNIST -> cSVHN same classes[B->B]:', Analytic.path_overlap(path2, path5))
    #print('Overlap MNIST -> cSVHN same classes[C->C]:', Analytic.path_overlap(path3, path6))

    pn_plotter = PathNetPlotter(pn)
    pn_plotter.plot_paths([path1, path2, path3, path4, path5, path6], filename='../logs/test_pdf')

#Analytic(pn).plot_training_counter()
