// Return a new array with each array being a chunk of the original array where size is the number of elements in each chunk.
export default <T>(array: T[], size: number): T[][] => {
	return array.reduce((resultArray: T[][], item, index) => {
		const chunkIndex = Math.floor(index / size);

		if(!resultArray[chunkIndex]) {
			resultArray[chunkIndex] = []; // start a new chunk
		}

		resultArray[chunkIndex].push(item);

		return resultArray;
	}, []);
};
