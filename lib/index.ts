#!/usr/bin/env node
import { S3, SageMaker } from "aws-sdk";
import s3uri_to_bucketkey from "./utils/s3uri_to_bucketkey";
import chunk_array from "./utils/chunk_array";
import * as util from "util";
import * as msRest from "@azure/ms-rest-js";
import * as TrainingApi from "@azure/cognitiveservices-customvision-training";
import * as inquirer from "inquirer";
import * as cliProgress from "cli-progress";
import * as path from "path";
import {promises as fs} from "fs";
const setTimeoutPromise = util.promisify(setTimeout);

(async () => {
	const {local, localPath} = await inquirer.prompt([
		{
			"type": "list",
			"name": "local",
			"message": "Do you want to use local files generated from GroundTruth or call the AWS services directly (remote)?",
			"choices": [
				"remote",
				"local"
			],
			"default": "remote"
		},
		{
			"type": "input",
			"name": "localPath",
			"message": "Enter the path to the local directory containing the GroundTruth files. This should contain the GroundTruth file (output.manifest) and the images.",
			"when": (answers) => answers.local === "local"
		}
	]);
	let awsRegion;
	if (local === "remote") {
		awsRegion = (await inquirer.prompt([
			{
				"type": "input",
				"name": "awsRegion",
				"message": "What AWS Region are you using?",
				"default": process.env.AWS_REGION || "us-east-1"
			}
		])).awsRegion;
	}
	const sageMaker = new SageMaker({"region": awsRegion});
	const labelingJobs = local === "remote" ? (await sageMaker.listLabelingJobs().promise()).LabelingJobSummaryList.filter((job) => job.LabelingJobStatus === "Completed") : [];
	const results = await inquirer.prompt([
		{
			"type": "list",
			"name": "sageMakerLabelingJob",
			"message": "Which Ground Truth labeling job would you like to use?",
			"choices": labelingJobs.map((job) => job.LabelingJobName),
			"when": () => local === "remote"
		},
		{
			"type": "input",
			"name": "customVisionTrainingKey",
			"message": "What is your Custom Vision Training Key?"
		},
		{
			"type": "input",
			"name": "customVisionTrainingEndpoint",
			"message": "What is your Custom Vision Training Endpoint?"
		},
		{
			"type": "list",
			"name": "automaticallyTrain",
			"message": "Would you like this tool run a training iteration for you after uploading your data?",
			"choices": ["Yes", "No"]
		},
		{
			"type": "number",
			"name": "confidenceThreshold",
			"message": "What should the confidence threshold be for importing tag into Custom Vision?",
			"default": 0.5
		},
		{
			"type": "number",
			"name": "chunkSize",
			"message": "How many items would you like to upload concurrently to Custom Vision?",
			"default": 64,
			"validate": (value) => {
				if (value > 64) {
					return "Chunk size has a limit of 64 items.";
				}

				return true;
			}
		}
	]);
	const labelingJob = labelingJobs.find((job) => job.LabelingJobName === results.sageMakerLabelingJob);
	if (!labelingJob && local === "remote") {
		throw new Error("Could not find labeling job.");
	}

	console.log("Retrieving AWS Ground Truth Manifest...");
	const s3OutputDatasetObject = local === "remote" ? s3uri_to_bucketkey(labelingJob.LabelingJobOutput.OutputDatasetS3Uri) : undefined;
	const s3Client = new S3({
		"region": awsRegion
	});
	const outputDataset = (local === "remote" ? (await s3Client.getObject(s3OutputDatasetObject).promise()).Body.toString() : await fs.readFile(path.join(localPath, "output.manifest"), "utf8")).split("\n").filter(Boolean).map((txt) => JSON.parse(txt));
	console.log(`Retrieved ${outputDataset.length} items from AWS Ground Truth Manifest...`);

	const defaultLabelingJobName = local === "remote" ? labelingJob.LabelingJobName : Object.keys(outputDataset[0]).find((key) => key !== "source-ref" && !key.endsWith("-metadata"));
	const labelingJobName = local === "remote" ? defaultLabelingJobName : ((await inquirer.prompt([
		{
			"type": "input",
			"name": "labelingJobName",
			"message": "What is the name of your labeling job?",
			"default": defaultLabelingJobName
		}
	])).labelingJobName);
	const publishIterationName: "detectModel" | "classifyModel" = await (async () => {
		const typesArray = outputDataset.map((item) => item[`${labelingJobName}-metadata`].type);
		const isEveryTypeIdentical = typesArray.every((type, i, array) => type === array[0]);
		if (!isEveryTypeIdentical) {
			throw new Error("Multiple types not supported.");
		}

		const type = typesArray[0];
		switch (type) {
		case "groundtruth/image-classification":
		case "groundtruth/image-classification-multilabel":
			return "classifyModel";
		case "groundtruth/object-detection":
			return "detectModel";
		default:
			throw new Error(`Unsupported type: ${type}.`);
		}
	})();
	const publishIterationNameFriendlyName = (() => {
		switch (publishIterationName) {
		case "detectModel":
			return "Object Detection";
		case "classifyModel":
			return "Classification";
		default:
			return "Unknown";
		}
	})();
	if (publishIterationNameFriendlyName !== "Unknown" && (await inquirer.prompt([
		{
			"type": "list",
			"name": "correctPublishIterationName",
			"choices": ["Yes", "No"],
			"message": `It looks like you are using a ${publishIterationNameFriendlyName} model. Is this correct?`
		}
	])).correctPublishIterationName === "No") {
		throw new Error("Unknown error retrieving type of model.");
	}

	const credentials = new msRest.ApiKeyCredentials({"inHeader": {"Training-key": results.customVisionTrainingKey}});
	const trainer = new TrainingApi.TrainingAPIClient(credentials, results.customVisionTrainingEndpoint);

	console.log("Creating/finding project...");
	const domains = (await trainer.getDomains()).filter((d) => {
		switch (publishIterationName) {
		case "detectModel":
			return d.type === "ObjectDetection";
		case "classifyModel":
			return d.type === "Classification";
		default:
			throw new Error(`Unknown domain type: ${d.type}`);
		}
	});
	const domainsIds = domains.map((d) => d.id);
	const existingProjects = await trainer.getProjects();
	const existingProjectChoices = existingProjects.filter((p) => domainsIds.includes(p.settings.domainId)).map((p) => p.name);
	const projectInput = (await inquirer.prompt([
		{
			"type": "list",
			"name": "shouldUseExistingProject",
			"message": "Should we use an existing Custom Vision project?",
			"choices": ["Yes", "No"],
			"validate": (input) => {
				if (input === "Yes") {
					return existingProjectChoices.length > 0 ? true : `There are no valid projects for ${publishIterationNameFriendlyName}. You must create a new project.`;
				}

				// We can always create a new project
				return true;
			}
		},
		{
			"type": "input",
			"name": "newProjectName",
			"message": "What should the name of your new project be?",
			"when": (answers) => answers.shouldUseExistingProject === "No",
			"default": labelingJobName
		},
		{
			"type": "list",
			"name": "domain",
			"choices": domains.map((d) => d.name),
			"message": "Which domain should we use to create your project?",
			"when": (answers) => answers.shouldUseExistingProject === "No"
		},
		{
			"type": "list",
			"name": "existingProjectName",
			"message": "What project would you like to use?",
			"choices": existingProjectChoices,
			"when": (answers) => answers.shouldUseExistingProject === "Yes",
			"default": existingProjectChoices.includes(labelingJobName) ? labelingJobName : existingProjectChoices[0]
		}
	]));
	let project: TrainingApi.TrainingAPIModels.Project | TrainingApi.TrainingAPIModels.CreateProjectResponse;
	if (projectInput.shouldUseExistingProject === "Yes") {
		project = existingProjects.find((p) => p.name === projectInput.existingProjectName);
		if (!project) {
			throw new Error(`Could not find project "${projectInput.existingProjectName}"`);
		}
		console.log("Project found...");
	} else {
		const domain = domains.find((domain) => domain.name === projectInput.domain);
		if (!domain) {
			throw new Error(`Could not find domain "${projectInput.domain}"`);
		}
		project = await trainer.createProject(projectInput.newProjectName, { "domainId": domain.id });
		console.log("Project created...");
	}

	console.log("Getting existing tags...");
	const tagHolder = new TagHolder(trainer, project);
	await tagHolder.loadExistingTags();

	const progress = new cliProgress.SingleBar({}, cliProgress.Presets.shades_classic);
	progress.start(outputDataset.length, 0);
	for (const outputChunk of chunk_array(outputDataset, results.chunkSize)) {
		// Wait one second to accommodate rate limit.
		await setTimeoutPromise(1000, null);

		const entries = await Promise.all(outputChunk.map(async (output) => {
			const outputImageURL = s3uri_to_bucketkey(output["source-ref"]);
			const outputImageURLParts = outputImageURL.Key.split("/");
			const imageName = outputImageURLParts[outputImageURLParts.length - 1];
			const outputImage = local === "remote" ? ((await s3Client.getObject(outputImageURL).promise()).Body as Buffer) : await fs.readFile(path.join(localPath, imageName));

			if (publishIterationNameFriendlyName === "Object Detection") {

				const annotations: any[] = output[labelingJobName].annotations;
				const classMap: {[key: string]: string} = output[`${labelingJobName}-metadata`]["class-map"];

				const imageSizeObject = output[labelingJobName]["image_size"];
				if (imageSizeObject.length !== 1) {
					throw new Error(`image-size has more than 1 item for ${output["source-ref"]}.`);
				}
				const imageSize = imageSizeObject[0];
				const regions = await Promise.all(annotations.map(async (annotation, index) => {
					const confidence = output[`${labelingJobName}-metadata`].objects[index].confidence;
					if (confidence < results.confidenceThreshold) {
						return null;
					}

					const className = classMap[annotation["class_id"]];
					const customVisionTag = await tagHolder.createOrRetrieve(className);
					if (!customVisionTag) {
						throw new Error(`Could not find tag for class "${className}"`);
					}

					return {
						"tagId": customVisionTag.id,
						"left": annotation.left / imageSize.width,
						"top": annotation.top / imageSize.height,
						"width": annotation.width / imageSize.width,
						"height": annotation.height / imageSize.height
					}
				}).filter(Boolean));

				const entry = {"name": imageName, "contents": outputImage, regions};
				return entry;
			} else if (publishIterationNameFriendlyName === "Classification") {
				let classMap: {[key: string]: number}; // key as the label name, and number as the confidence level
				if (output[`${labelingJobName}-metadata`]["class-name"]) {
					classMap = {
						[output[`${labelingJobName}-metadata`]["class-name"]]: output[`${labelingJobName}-metadata`].confidence
					};
				} else if (output[`${labelingJobName}-metadata`]["class-map"]) {
					classMap = Object.entries(output[`${labelingJobName}-metadata`]["class-map"]).reduce((obj, classDetails: [string, string]) => {
						const [key, value] = classDetails;
						obj[value] = output[`${labelingJobName}-metadata`]["confidence-map"][key];
						return obj;
					}, {});
				}

				return {
					outputImage,
					"options": {
						"tagIds": await Promise.all(Object.entries(classMap).filter(([, confidence]) => confidence >= results.confidenceThreshold).map(async ([className]) => {
							return (await tagHolder.createOrRetrieve(className)).id;
						}))
					}
				};
			} else {
				throw new Error(`Unknown publishIterationName "${publishIterationNameFriendlyName}"`);
			}
		}));

		switch (publishIterationNameFriendlyName) {
		case "Object Detection":
			const batch: TrainingApi.TrainingAPIModels.ImageFileCreateBatch = {"images": entries as TrainingApi.TrainingAPIModels.ImageFileCreateEntry[]};
			await trainer.createImagesFromFiles(project.id, batch);
			break;
		case "Classification":
			await Promise.all(entries.map(async (entry: {"outputImage": Buffer, options: {"tagIds": string[]}}) => {
				await trainer.createImagesFromData(project.id, entry.outputImage, entry.options);
			}));
			break;
		default:
			throw new Error(`Unknown publishIterationName "${publishIterationNameFriendlyName}"`);
		}

		progress.increment(entries.length);
	}
	progress.stop();

	// Training
	if (results.automaticallyTrain === "Yes") {
		console.log("Training...");
		let trainingIteration = await trainer.trainProject(project.id);

		// Wait for training to complete
		console.log("Training started (this might take a while)...");
		while (trainingIteration.status === "Training") {
			// Wait for ten seconds
			await setTimeoutPromise(10000, null);
			trainingIteration = await trainer.getIteration(project.id, trainingIteration.id);
		}
		console.log(`Training Complete with Status: ${trainingIteration.status}...`);
	}

	console.log("✅ Done!");
})();

interface PromiseHandler<T> {
	"resolve": (value: T | PromiseLike<T>) => void;
	"reject": (reason?: any) => void;
}

class TagHolder {
	trainer: TrainingApi.TrainingAPIClient;
	project: TrainingApi.TrainingAPIModels.Project;
	tags: Map<string, TrainingApi.TrainingAPIModels.CreateTagResponse | TrainingApi.TrainingAPIModels.Tag>;

	#pendingTags: Map<string, PromiseHandler<TrainingApi.TrainingAPIModels.CreateTagResponse | TrainingApi.TrainingAPIModels.Tag>[]>;

	constructor(trainer: TrainingApi.TrainingAPIClient, project: TrainingApi.TrainingAPIModels.Project) {
		this.tags = new Map<string, TrainingApi.TrainingAPIModels.CreateTagResponse | TrainingApi.TrainingAPIModels.Tag>();
		this.#pendingTags = new Map();
		this.trainer = trainer;
		this.project = project;
	}

	async createOrRetrieve(tag: string): Promise<TrainingApi.TrainingAPIModels.CreateTagResponse | TrainingApi.TrainingAPIModels.Tag> {
		if (this.tags.has(tag)) {
			return this.tags.get(tag);
		}

		if (!this.#pendingTags.has(tag)) {
			this.#pendingTags.set(tag, []);
			const tagResponse = await this.trainer.createTag(this.project.id, tag);
			this.tags.set(tag, tagResponse);
			this.#pendingTags.get(tag).forEach(handler => handler.resolve(tagResponse));
			return tagResponse;
		} else {
			return new Promise((resolve, reject) => {
				this.#pendingTags.get(tag).push({
					resolve,
					reject
				});
			});
		}
	}

	async loadExistingTags(): Promise<void> {
		const existingTags = await this.trainer.getTags(this.project.id);
		existingTags.filter((tag) => tag.type === "Regular").forEach((tag) => this.tags.set(tag.name, tag));
	}
}
