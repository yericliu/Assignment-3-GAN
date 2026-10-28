import torch


def train_model(model, data_loader, criterion, optimizer, device="cpu", epochs=10):
    model.to(device)
    history = []
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
        row = {"epoch": epoch + 1, "train_loss": total_loss / total,
               "train_accuracy": correct / total}
        history.append(row)
        print(f"Epoch {epoch + 1}/{epochs}: loss={row['train_loss']:.4f}, "
              f"train accuracy={row['train_accuracy']:.2%}")
    model.training_history = history
    return model


def evaluate_model(model, data_loader, criterion, device="cpu"):
    model.to(device)
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return {"test_loss": total_loss / total, "test_accuracy": correct / total,
            "test_samples": total}



def train_gan(model, data_loader, criterion, optimizer, device='cpu', epochs=10):
    generator, discriminator = model
    optimizer_G, optimizer_D = optimizer
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        for real_images, _ in data_loader:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            noise = torch.randn(batch_size, 100, device=device)
            fake_images = generator(noise)

            optimizer_D.zero_grad()
            loss_real = criterion(discriminator(real_images), real_labels)
            loss_fake = criterion(discriminator(fake_images.detach()), fake_labels)
            loss_D = loss_real + loss_fake
            loss_D.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()
            loss_G = criterion(discriminator(fake_images), real_labels)
            loss_G.backward()
            optimizer_G.step()

        print(f'Epoch {epoch + 1}: D loss = {loss_D.item():.4f}, G loss = {loss_G.item():.4f}')

    return generator, discriminator
