import { URL } from "url";

export default (url: string): { "Bucket": string; "Key": string } => {
	const s3uri = new URL(url);
	const bucket = s3uri.hostname;
	const key = s3uri.pathname.substr(1);

	return {
		"Bucket": bucket,
		"Key": key
	};
};
