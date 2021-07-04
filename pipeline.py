import valohai


def main(old_config):
    papi = valohai.Pipeline(name="train", config=old_config)

    # Define nodes
    preprocess = papi.execution("convert-superbai", "preprocess")
    weights = papi.execution("weights")
    train = papi.execution("train")
    evaluate = papi.execution("detect", "evaluate")

    # Configure pipeline
    preprocess.output("classes.txt").to(train.input("classes"))
    preprocess.output("train/*").to(train.input("train"))
    preprocess.output("test/*").to(train.input("test"))
    preprocess.output("classes.txt").to(evaluate.input("classes"))

    weights.output("model/*").to(train.input("model"))

    train.output("model/*").to(evaluate.input("model"))

    return papi
