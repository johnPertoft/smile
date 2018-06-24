import * as tf from '@tensorflow/tfjs'


export class Model {
    constructor() {}

    async load() {
        this.model = await tf.loadFrozenModel(
            'models/test/tensorflowjs_model.pb',
            'models/test/weights_manifest.json')
    }

    dispose() {
        if (this.model) {
            this.model.dispose()
        }
    }

    async translate(input) {
        const x = input
            .asType('float32')
            .div(tf.scalar(255.0 / 2))
            .sub(tf.scalar(1.0))
            .reshape([1, ...input.shape])

        const input_node_name = 'todo'
        const output_node_name = 'todo'

        return this.model.execute({[input_node_name]: x}, output_node_name)
    }
}
