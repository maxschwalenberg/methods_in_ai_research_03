def show_results(model):
    model.create_bag_of_words()
    model.fit()
    model.predict()
    model.evaluate()

    print(f"{model.__class__.__name__}:")
    print("Accuracy:", model.accuracy)
    print("Precision:", model.precision)
    print("Recall:", model.recall)
    for label, metrics in model.class_metrics.items():
        print(f"{label}:")
        print(f"  Precision: {metrics['Precision']:.4f}")
        print(f"  Recall: {metrics['Recall']:.4f}")
