import { Component, ViewChild, ElementRef, OnInit } from '@angular/core';
import * as tf from '@tensorflow/tfjs';
import { LESCLASSES } from '../assets/imagenet-classes';

const TAILLE_IMAGE  = 224;
const TOPK_PREDICTIONS = 5;


@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})


export class AppComponent implements OnInit {
  model: tf.Model;
  classes: any[];
  imageData: ImageData;
  title = 'TensorflowPFF';

  @ViewChild("chosenImage",{static: true}) img: ElementRef;
  @ViewChild("fileUpload",{static: true}) fileUpload: ElementRef;

  ngOnInit() {
    this.loadModel();
  }

  async loadModel() {
    this.model = await tf.loadModel("../assets/model.json");
  }

  fileChangeEvent(event: any) {
    const file = event.target.files[0];
    if (!file || !file.type.match("image.*")) {
      return;
    }

    this.classes = [];

    const reader = new FileReader();
      reader.onloadend = e => {
      this.img.nativeElement.src = e.target["result"];
      this.predict(this.img.nativeElement);
    };

    reader.onload = e => {
      this.img.nativeElement.src = e.target["result"];
      this.predict(this.img.nativeElement);
    };
    reader.readAsDataURL(file);
  }

async predict(imageData: ImageData): Promise<any> {
  this.fileUpload.nativeElement.value = '';
  const startTime = performance.now();
  const logits = tf.tidy(() => {
    // tf.fromPixels() renvoie un tenseur à partir d'un élément d'image.
    const img = tf.fromPixels(imageData).toFloat();

    const offset = tf.scalar(127.5);
    // Normaliser l'image de [0, 255] à [-1, 1].
    const normalized = img.sub(offset).div(offset);

    // Transformez en un élément pour pouvoir le prédire.
    const batched = normalized.reshape([1, TAILLE_IMAGE , TAILLE_IMAGE , 3]);

    // Faites une prédiction via mobilenet.
    return this.model.predict(batched);
  });

  // Convertissez les logits en probabilités et en noms de classes.
  this.classes = await this.getTopKClasses(logits, TOPK_PREDICTIONS);
  const totalTime = performance.now() - startTime;
  console.log(`Fait en ${Math.floor(totalTime)}ms`);
}

  async getTopKClasses(logits, topK): Promise<any[]> {
    const values = await logits.data();

    const valuesAndIndices = [];
    for (let i = 0; i < values.length; i++) {
      valuesAndIndices.push({ value: values[i], index: i });
    }
    valuesAndIndices.sort((a, b) => {
      return b.value - a.value;
    });
    const topkValues = new Float32Array(topK);
    const topkIndices = new Int32Array(topK);
    for (let i = 0; i < topK; i++) {
      topkValues[i] = valuesAndIndices[i].value;
      topkIndices[i] = valuesAndIndices[i].index;
    }

    const topClassesAndProbs = [];
    for (let i = 0; i < topkIndices.length; i++) {
      topClassesAndProbs.push({
        className: LESCLASSES[topkIndices[i]],
        probability: topkValues[i]
      });
    }
    return topClassesAndProbs;
  }
}

