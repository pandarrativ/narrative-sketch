class MyImage {
  constructor() {
    this.base64 = ''
  }
  async readFromBlob(blob) {
    const reader = new FileReader()
    reader.readAsDataURL(blob.slice(0, blob.size, 'image/png'))
    async function readerOnLoad() {
      let result
      await new Promise((resolve) => {
        reader.onloadend = () => {
          resolve(reader.result)
        }
      }).then((res) => {
        result = res
      })
      return result
    }
    this.base64 = await readerOnLoad()
  }
}

export default MyImage
