import * as tf from '@tensorflow/tfjs';
import {Model} from './model';


window.onload = async () => {
    const imgs = document.getElementsByTagName('img')

    const model = new Model()
    console.time('Loading of model');
    await model.load()
    console.timeEnd('Loading of model')

    const X = tf.fromPixels(imgs[0])

    console.time('First translation')
    const translation = model.translate(X)
    console.timeEnd('First translation')

    translation.print()

    model.dispose()
}
