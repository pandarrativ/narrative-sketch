import * as d3 from 'd3'
import MyImage from './Image'
// this file is used to wirte server communication functions
// such as evaluate a sketching, read image from server ....
export async function evaluate(sketchingImage) {
  await new Promise((r) => setTimeout(r, 500))
  return Math.random() * 100
}

const AIImageCache = {}
export async function fetchAIImage(img, step) {
  if (img in AIImageCache) {
    if (step !== undefined) {
      return AIImageCache[img][step]
    } else {
      return AIImageCache[img]
    }
  } else {
    //! hard-code
    //! please change these code to communicate with a real server
    img = 'owl'
    const list = []
    const steps = d3.range(17, 81)
    const urls = steps.map((step) => `/images/sketchImg/Incubation/im${step}.png`)
    const results = {}

    urls.forEach(function (url, i) {
      // (1)
      list.push(
        // (2)
        fetch(url, {
          headers: {
            'Content-Type': 'image/png'
            // 'Content-Type': 'application/x-www-form-urlencoded',
          }
        })
          .then((response) => response.blob())
          .then(async function (blob) {
            const myImage = new MyImage()
            await myImage.readFromBlob(blob)
            results[steps[i]] = myImage
          })
      )
    })

    await Promise.all(list) // (4)

    AIImageCache[img] = results
    if (step !== undefined) {
      return results[step]
    } else {
      return results
    }
  }
}

/**
 * in the inspiration combination step,
 * find relevent objects of the word
 * @param {*} word
 */
export async function fetchRelevantObjects(word) {
  await new Promise((r) => setTimeout(r, 500))
  return [
    {
      id: 'eagle',
      label: 'Eagle'
    },
    {
      id: 'bird',
      label: 'Bird'
    },
    {
      id: 'flower',
      x: 0,
      y: 80,
      label: 'Flower'
    },
    {
      id: 'solemn',
      label: 'Sloemn'
    },
    {
      id: 'flying',
      label: 'Flying'
    }
  ]
}
