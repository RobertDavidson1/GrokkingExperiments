# https://typer.tiangolo.com/tutorial/package/#create-a-project
import typer



# Show help message if no arg given
# https://typer.tiangolo.com/tutorial/commands/#show-the-help-message-if-no-command-is-given
app = typer.Typer(no_args_is_help=True)

@app.command()
def train():
    print("Starting training")


@app.command()
def goodbye():
    print("Goodbye ")