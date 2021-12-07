<script>
  import ChooseModelButton from "./ChooseModelButton.svelte";
  import ModelColumn from "./ModelColumn.svelte";
</script>

<head>
  <meta
    http-equiv="Cache-Control"
    content="no-cache, no-store, must-revalidate"
  />
  <meta http-equiv="Pragma" content="no-cache" />
  <meta http-equiv="Expires" content="0" />
  <div class="content" /></head
>

<div class="content" id="dank">
  <h1 id = "title">Face Morph Madness </h1>

  <h2 id="promo"> You've seen the faces. Now watch them morph into each other. </h2>

  <div class="model-columns">
    <div>
      <ModelColumn
        class="model"
        modelName="DCGAN (2017)"
        id="dcgan"
        buttonText="Get a new Interpolation"
      />
    </div>
    <div>
      <ModelColumn class="model" modelName="StyleGAN 2 (2020)" id="stylegan2" buttonText="Get a new Interpolation" />
    </div>
    <div>
      <ModelColumn class="model" modelName="StyleGAN (2019)" id="stylegan" buttonText="Get a new Interpolation"/>
    </div>
  </div>
</div>

<style global lang="scss">
  @use "@material/button/styles";
  @use "@material/data-table/data-table";
  @import url('https://fonts.googleapis.com/css2?family=Bangers&display=swap');


  #title {
    font-size: 4rem; 
    font-family: "Bangers";
  }

  #promo {
    font-size: 2rem;
  }
  .content {
    text-align: center;
  }

  .model-columns {
    display: flex;
    flex-direction: row;
    flex-wrap: wrap;
    justify-content: center;
    gap: 1rem;
  }
</style>
