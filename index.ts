import * as tf from '@tensorflow/tfjs-core'

declare const window: any

//const optimizer = tf.train.sgd(0.001)
const optimizer = tf.train.adam(0.001, 0.9, 0.999, 1e-8)

const batchSize = 1
const inputSize = 320

const inputChannels = 3
const outputChannels = 128
const strides = [1, 1] as [number, number]
const padding = 'valid'

const depthwiseFilter = tf.variable(tf.randomNormal([3, 3, inputChannels, 1])) as tf.Tensor4D
const pointwiseFilter = tf.variable(tf.randomNormal([1, 1, inputChannels, outputChannels])) as tf.Tensor4D
const bias = tf.variable(tf.zeros([outputChannels])) as tf.Tensor1D

const factor = tf.scalar(0.10000000149011612)

const input = tf.randomNormal([batchSize, inputSize, inputSize, inputChannels]) as tf.Tensor4D

function forward(x: tf.Tensor4D, withMaximum: boolean): tf.Tensor4D {
  return tf.tidy(() => {
    let y = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]]) as tf.Tensor4D
    y = tf.separableConv2d(y, depthwiseFilter, pointwiseFilter, strides, padding)
    y = tf.add(y, bias)

    const min = tf.mul(y, factor)

    // this op seems to be the issue when training
    if (withMaximum) {
      y = tf.maximum(y, min)
    }

    return y
  })
}

function train(x: tf.Tensor4D, withMaximum: boolean) {
  return optimizer.minimize(() => {
    const out = forward(x, withMaximum)
    return tf.sum(out)
  }, true)
}

window.printMemory = function() {
  console.log(tf.memory())
}

window.runForward = function(iterations: number) {
  for (let i = 0; i < iterations; i++) {
    const out = forward(input, true)
    out.dispose()
  }
}

window.runTrain = function(iterations: number, withMaximum: boolean) {
  for (let i = 0; i < iterations; i++) {
    const loss = train(input, withMaximum)
    if (loss) {
      loss.dispose()
    }
  }
}