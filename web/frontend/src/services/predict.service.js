import createApiClient from "./api.service";

class PredictService {
    constructor(baseUrl = "/api/detect/") {
        this.api = createApiClient(baseUrl);
    }
    async uploadImage(data) {
        const respone = await this.api.post("/", data);
        return respone;
    }
}
export default new PredictService();