<template>
    <div class="container">
        <h3>Leaf Classification</h3>
        <div class="formInput">
            <div class="form-box">
                <form @submit.prevent="uploadImage">
                    <label for="imageInput" style="margin-right:5px;"><b>Upload an image here to identify:</b></label>
                    <input class="input-button" type="file" id="imageInput" @change="onFileChange" />
                    <button class="submit-button" type="submit" :disabled="!selectedFile">Upload</button>
                </form>
                <div v-if="loading">Loading...</div>
                <div v-else-if="error">{{ error }}</div>
            </div>
            <hr>
            <div v-if="imageUrl" class="image-container">
                <h4>Uploaded Image:</h4>
                <img :src="imageUrl" alt="Uploaded Image" style="width:400px; height:auto;" />
                <ul v-if="diseases.length">
                    <li v-for="(disease, index) in diseases" :key="index">
                        <p style="font-size:20px;"><b>{{ index + 1 }}. {{ disease }}</b></p>
                    </li>
                </ul>
            </div>
        </div>
    </div>
</template>

<script>
import PredictService from "../services/predict.service";

export default {
    components: {
    },
    data() {
        return {
            diseases: [],
            loading: false,
            error: null,
            selectedFile: null,
            imageUrl: null,
        };
    },
    methods: {
        onFileChange(event) {
            this.diseases = [];
            this.selectedFile = event.target.files[0];
            this.imageUrl = URL.createObjectURL(this.selectedFile); 
        },
        async uploadImage() {
            if (!this.selectedFile) {
                this.error = 'Please select an image file';
                return;
            }

            this.loading = true;
            this.error = null;

            const formData = new FormData();
            formData.append('image_input', this.selectedFile);
            try {
                const response = await PredictService.uploadImage(formData);
                // this.diseases = response.data;
                this.diseases.push(response.data.output);
                console.log(response.data)
            } catch (error) {
                this.error = 'Failed to upload image or load diseases';
                console.error(error);
            } finally {
                this.loading = false;
            }
        },

    }
};
</script>

<style scoped>


.image-container {
    display: flex;
    flex-direction: column;
    align-items: center; /* Center content horizontally */
    margin-top: 20px;
}

.image-container img {
    margin-bottom: 20px;
}

ul {
    list-style-type: none;
    padding: 0;
    text-align: center; /* Center text */
}

.form-box {
    border: 1px solid #ccc;
    padding: 30px;
    border-radius: 8px;
    margin-bottom: 16px;
    background-color: #f9f9f9;

}

.submit-button {
    -moz-box-shadow: inset 0px 1px 0px 0px #ffffff;
    -webkit-box-shadow: inset 0px 1px 0px 0px #ffffff;
    box-shadow: inset 0px 1px 0px 0px #ffffff;
    background: -webkit-gradient(linear, left top, left bottom, color-stop(0.05, #ededed), color-stop(1, #dfdfdf));
    background: -moz-linear-gradient(center top, #ededed 5%, #dfdfdf 100%);
    filter: progid:DXImageTransform.Microsoft.gradient(startColorstr='#ededed', endColorstr='#dfdfdf');
    background-color: #ededed;
    -webkit-border-top-left-radius: 23px;
    -moz-border-radius-topleft: 23px;
    border-top-left-radius: 23px;
    -webkit-border-top-right-radius: 23px;
    -moz-border-radius-topright: 23px;
    border-top-right-radius: 23px;
    -webkit-border-bottom-right-radius: 23px;
    -moz-border-radius-bottomright: 23px;
    border-bottom-right-radius: 23px;
    -webkit-border-bottom-left-radius: 23px;
    -moz-border-radius-bottomleft: 23px;
    border-bottom-left-radius: 23px;
    text-indent: 0;
    border: 1px solid #dcdcdc;
    display: inline-block;
    color: #130707;
    font-family: Arial;
    font-size: 16px;
    font-weight: bold;
    font-style: normal;
    height: 32px;
    line-height: 32px;
    width: 104px;
    text-decoration: none;
    text-align: center;
    text-shadow: 1px 1px 0px #ffffff;
}

.submit-button:hover {
    background: -webkit-gradient(linear, left top, left bottom, color-stop(0.05, #dfdfdf), color-stop(1, #ededed));
    background: -moz-linear-gradient(center top, #dfdfdf 5%, #ededed 100%);
    filter: progid:DXImageTransform.Microsoft.gradient(startColorstr='#dfdfdf', endColorstr='#ededed');
    background-color: #dfdfdf;
}

.submit-button:active {
    position: relative;
    top: 1px;
}

.container {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 20px;
}

.formInput {
    width: 100%;
    /* max-width: 800px; */
    margin-bottom: 20px;
}

ul {
    list-style-type: none;
    padding: 0;
}

li {
    margin-bottom: 20px;
}
</style>
