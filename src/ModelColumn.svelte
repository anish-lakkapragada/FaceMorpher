<script> 
    import ChooseModelButton from "./ChooseModelButton.svelte";
    export let modelName; 
    export let id; 
    export let buttonText;
    let video_path; 
    fetch(`https://face-morpher-api.loca.lt/${id}`).then(response => response.text()).then((response) => {
        console.log("this is response : " + response); 
        video_path = `serve/${id}/` + response; 
        console.log(`this is video_path : ${video_path}`)
    }) 
</script>

<div class="divvy"> 
    <div style="display: flex; flex-direction: column;"> 
        <h2> {modelName} </h2>
        
        <ChooseModelButton id={id} text={buttonText}></ChooseModelButton>
        
        <!-- have a video here, maybe actually describe how the model works as -->

        <video controls name="Video Name">
            <source src={video_path}>
            <track kind="captions"> 
          </video>

    </div>
</div>

<style> 
    *{
    margin-top: 0; 
    }
</style>